FROM python:3.11-slim

ARG HF_TOKEN
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    TORCH_HOME=/models \
    OMP_NUM_THREADS=4 \
    HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}

RUN apt-get update && \
    apt-get install -y --no-install-recommends libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download the model at build time (~885MB)
# HUGGING_FACE_HUB_TOKEN env var is auto-detected by huggingface_hub
RUN python3 -c "from transformers import AutoModelForImageSegmentation; \
    AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True); \
    print('Model downloaded successfully')"

COPY app ./app

RUN mkdir -p /models
VOLUME ["/models"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
