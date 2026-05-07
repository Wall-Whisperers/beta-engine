# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

RUN groupadd --system appgroup \
 && useradd --system --gid appgroup --home-dir /app appuser \
 && mkdir -p /data/walls /data/runs \
 && chown -R appuser:appgroup /data

COPY requirements.txt ./

# Build tools needed by pymunk (compiles a C extension).
# Removed after install to keep the image lean.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libc6-dev \
 && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first so the default index doesn't pull the
# 2.5 GB CUDA build. Subsequent pip calls see torch already satisfied.
RUN pip install --no-cache-dir \
        torch \
        --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"

CMD ["python", "-m", "grid_editor.server"]
