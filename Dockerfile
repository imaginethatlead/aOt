FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

ENV MODEL_ID=Qwen/Qwen2.5-Omni-7B
ENV USE_AUDIO_IN_VIDEO=true
ENV MAX_NEW_TOKENS=2048
ENV VIDEO_MAX_PIXELS=20070400

CMD ["python", "-u", "/app/handler.py"]
