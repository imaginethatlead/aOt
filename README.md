# Runpod Avocado Worker

This folder is a standalone worker repo for Runpod Serverless.
It downloads a video URL, runs Avocado (Qwen2.5-Omni), and returns caption text.

## Input contract

Runpod input JSON:

```json
{
  "video_url": "https://.../video.mp4",
  "model_id": "Qwen/Qwen2.5-Omni-7B",
  "prompt": "Give a detailed audio-visual analysis of this video."
}
```

Only `video_url` is required.

## Output contract

```json
{
  "text": "caption text",
  "model_id": "Qwen/Qwen2.5-Omni-7B",
  "prompt": "Give a detailed audio-visual analysis of this video.",
  "timing_s": 12.345
}
```

Runpod wraps this under `output`.

## Build and push

Example with Docker Hub:

```bash
docker build -t <docker-user>/aot-avocado-worker:latest .
docker push <docker-user>/aot-avocado-worker:latest
```

Then in Runpod:
- Container image: `<docker-user>/aot-avocado-worker:latest`
- Container start command: leave blank (Dockerfile sets CMD)

## Baking the model

To bake the model into the image, pass a build arg:

```bash
docker build --build-arg HF_TOKEN=YOUR_TOKEN -t <docker-user>/aot-avocado-worker:latest .
```

If `HF_TOKEN` is not set, the model will download at runtime.

## Environment variables

- `MODEL_ID` (default: `Qwen/Qwen2.5-Omni-7B`)
- `DEFAULT_PROMPT` (default: detailed A/V prompt)
- `USE_AUDIO_IN_VIDEO` (default: true)
- `MAX_NEW_TOKENS` (default: 2048)
- `VIDEO_MAX_PIXELS` (default: 20070400)
- `ATTN_IMPL` (optional, e.g. `flash_attention_2`)
