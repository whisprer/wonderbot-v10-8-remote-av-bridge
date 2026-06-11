FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive     PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PIP_NO_CACHE_DIR=1     HF_HOME=/srv/weights/wonderbot/hf-cache     TRANSFORMERS_CACHE=/srv/weights/wonderbot/hf-cache     BITSANDBYTES_NOWELCOME=1

RUN apt-get update && apt-get install -y --no-install-recommends     python3 python3-pip python3-venv python3-dev     git curl wget ca-certificates ffmpeg libsndfile1 libgl1 jq tini     build-essential &&     rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN python3 -m pip install --upgrade pip wheel setuptools &&     python3 -m pip install --index-url https://download.pytorch.org/whl/cu126 torch torchaudio &&     python3 -m pip install -e .[server-hf,live-full,hf-voice,voice,dev]

RUN chmod +x /app/scripts/server/docker-entrypoint.sh

ENTRYPOINT ["/usr/bin/tini", "--", "/app/scripts/server/docker-entrypoint.sh"]
CMD ["wonderbot", "--config", "/app/configs/profiles/dual-p40-server-qwen14b.toml", "--diagnostics"]
