FROM pytorch/pytorch:2.6.0-cuda12.1-cudnn9-runtime

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

ENV MODEL_ID=Qwen/Qwen2.5-Omni-7B
ENV USE_AUDIO_IN_VIDEO=true
ENV MAX_NEW_TOKENS=2048
ENV VIDEO_MAX_PIXELS=20070400

# Optional: bake the model into the image to avoid runtime downloads.
# Set HF_TOKEN as a build arg in the Runpod UI.
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
ENV HF_HOME=/models/hf
RUN if [ -n "$HF_TOKEN" ]; then \
    printf '%s\n' \
    'from huggingface_hub import snapshot_download' \
    'import os' \
    'model_id = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-Omni-7B")' \
    'snapshot_download(repo_id=model_id, local_dir=os.environ.get("HF_HOME", "/models/hf"), local_dir_use_symlinks=False)' \
    > /tmp/download_model.py && \
    python /tmp/download_model.py && \
    rm /tmp/download_model.py; \
  else echo "HF_TOKEN not set; model will download at runtime."; fi

CMD ["python", "-u", "/app/handler.py"]
