# Torch pip wheels bundle their own CUDA/cuDNN libraries, so a plain Python base
# suffices — GPU use only needs the NVIDIA driver + container toolkit on the host.
# ML_EXTRAS picks the engine tiers baked into the image:
#   "diarize align local" (default) supports every TOLKA_ENGINE value;
#   "diarize align" for hybrid/remote-only deployments (smaller, no easytranscriber);
#   "diarize" for remote-only.
ARG TORCH_VARIANT=gpu
ARG ML_EXTRAS="diarize align local"

FROM ghcr.io/astral-sh/uv:0.8.20 AS uvbin

FROM python:3.12-slim-bookworm
ARG TORCH_VARIANT
ARG ML_EXTRAS

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uvbin /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_LINK_MODE=copy

# Dependency layer: cached until pyproject/lock change
COPY pyproject.toml uv.lock README.md ./
# The cache mount keeps uv's unpacked-wheel cache out of the image layers;
# without it the layer carries a near-duplicate of the entire venv (~8GB for gpu)
RUN --mount=type=cache,target=/root/.cache/uv \
    EXTRA_FLAGS="--extra ${TORCH_VARIANT}"; \
    for extra in ${ML_EXTRAS}; do EXTRA_FLAGS="${EXTRA_FLAGS} --extra ${extra}"; done; \
    uv sync --frozen --no-dev --no-install-project ${EXTRA_FLAGS}

COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    EXTRA_FLAGS="--extra ${TORCH_VARIANT}"; \
    for extra in ${ML_EXTRAS}; do EXTRA_FLAGS="${EXTRA_FLAGS} --extra ${extra}"; done; \
    uv sync --frozen --no-dev ${EXTRA_FLAGS}

ENV PATH=/app/.venv/bin:$PATH \
    HF_HOME=/models \
    TOLKA_MODEL_CACHE_DIR=/models \
    TOLKA_WORK_DIR=/data/work \
    TOLKA_DB_PATH=/data/tolka.sqlite3

# /app (venv included) stays root-owned read-only; chowning it would copy the
# whole venv into a new layer. The runtime user only writes /models and /data.
RUN mkdir -p /models /data && chown -R 1000:1000 /models /data
USER 1000:1000

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/readyz', timeout=3)"
CMD ["uvicorn", "tolka.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
