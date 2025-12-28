import os
import tempfile
import time
from pathlib import Path

import requests
import runpod
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info


MODEL = None
PROCESSOR = None
LOADED_MODEL_ID = None


DEFAULT_PROMPT = os.environ.get(
    "DEFAULT_PROMPT",
    "Give a detailed audio-visual analysis of this video."
)
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-Omni-7B")
USE_AUDIO_IN_VIDEO = os.environ.get("USE_AUDIO_IN_VIDEO", "true").lower() == "true"
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "2048"))
VIDEO_MAX_PIXELS = int(os.environ.get("VIDEO_MAX_PIXELS", "20070400"))
ATTN_IMPL = os.environ.get("ATTN_IMPL")


def download_video(url: str) -> Path:
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    fd, temp_path = tempfile.mkstemp(suffix=".mp4")
    with os.fdopen(fd, "wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    return Path(temp_path)


def load_model(model_id: str):
    global MODEL, PROCESSOR, LOADED_MODEL_ID
    if MODEL is not None and PROCESSOR is not None and LOADED_MODEL_ID == model_id:
        return MODEL, PROCESSOR

    kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if ATTN_IMPL:
        kwargs["attn_implementation"] = ATTN_IMPL

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_id, **kwargs)
    model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained(model_id)

    MODEL = model
    PROCESSOR = processor
    LOADED_MODEL_ID = model_id
    return MODEL, PROCESSOR


def generate_caption(model, processor, file_path: str, prompt: str) -> str:
    conversation = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
                        "capable of perceiving auditory and visual inputs, as well as generating text and speech."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": file_path,
                    "max_pixels": VIDEO_MAX_PIXELS,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        },
    ]

    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(
        conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO
    )
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
    )
    inputs = inputs.to(model.device).to(model.dtype)
    text_ids = model.generate(
        **inputs,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
        do_sample=False,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    generated_ids = text_ids[:, inputs.input_ids.shape[1] :]
    output = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return output.strip()


def handler(event: dict) -> dict:
    input_data = event.get("input", {})
    video_url = input_data.get("video_url") or input_data.get("url")
    if not video_url:
        return {"error": "Missing video_url in input."}

    model_id = input_data.get("model_id") or MODEL_ID
    prompt = input_data.get("prompt") or DEFAULT_PROMPT

    model, processor = load_model(model_id)
    start = time.time()

    video_path = download_video(video_url)
    try:
        text = generate_caption(model, processor, str(video_path), prompt)
    finally:
        try:
            video_path.unlink()
        except OSError:
            pass

    elapsed_s = round(time.time() - start, 3)
    return {
        "text": text,
        "model_id": model_id,
        "prompt": prompt,
        "timing_s": elapsed_s,
    }


runpod.serverless.start({"handler": handler})
