# A stable, real base image that exists.
# We then install PyTorch ourselves (2.6.0 + CUDA 12.4 wheels).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

# System dependencies:
# - ffmpeg: video/audio decoding backend (fixes audioread NoBackendError)
# - libsndfile1: required by pysoundfile/soundfile used by librosa
# - python3/pip: runtime
# - git/ca-certificates/curl: useful and often needed by HF downloads
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-pip \
        ffmpeg libsndfile1 \
        git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Use python/pip consistently
RUN python3 -m pip install --upgrade pip

# Install PyTorch 2.6.0 with CUDA 12.4 wheels (works well with 4090).
# Note: This avoids relying on non-existent pytorch/pytorch Docker tags.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# Python dependencies for your worker
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Worker code
COPY handler.py /app/handler.py

# Runtime defaults (can be overridden in Runpod UI)
ENV MODEL_ID=Qwen/Qwen2.5-Omni-7B
ENV USE_AUDIO_IN_VIDEO=true
ENV MAX_NEW_TOKENS=2048
ENV VIDEO_MAX_PIXELS=20070400

# Optional: bake the model into the image to avoid runtime downloads.
# Set HF_TOKEN as a build arg in the Runpod UI (or your CI).
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
    python3 /tmp/download_model.py && \
    rm /tmp/download_model.py; \
  else echo "HF_TOKEN not set; model will download at runtime."; fi

CMD ["python3", "-u", "/app/handler.py"]
