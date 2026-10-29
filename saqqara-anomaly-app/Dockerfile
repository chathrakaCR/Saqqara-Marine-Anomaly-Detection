# Marine anomaly detection app (P000285DS) - CPU image
#
#   docker build -t marine-anomaly .
#   docker run -p 8000:8000 -v "$PWD/data:/data" marine-anomaly            # web app on http://localhost:8000
#   docker run -v "$PWD/data:/data" marine-anomaly python -m app.cli -i /data/flight.mp4 -o /data/results/flight
#
# Build arguments:
#   SKLEARN_VERSION   must match the scikit-learn version that saved models/track_a_model.pkl
#   TORCH_INDEX_URL   where to get PyTorch (default: the small CPU-only build)
ARG BASE_IMAGE=python:3.11-slim-bookworm
FROM ${BASE_IMAGE}

ARG SKLEARN_VERSION=1.6.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATA_DIR=/data \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --index-url "${TORCH_INDEX_URL}" torch \
 && pip install -r requirements.txt "scikit-learn==${SKLEARN_VERSION}"

COPY app ./app
COPY config ./config
COPY tools ./tools
COPY models ./models

RUN mkdir -p /data

EXPOSE 8000
VOLUME ["/data"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS http://localhost:8000/api/health || exit 1
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
