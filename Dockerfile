# CUDA runtime with cuDNN: ctranslate2 dlopens system cuDNN/cuBLAS and does not
# reuse torch's bundled pip wheels — the plain (non-cudnn) runtime image breaks it.
ARG TORCH_VARIANT=gpu

FROM ghcr.io/astral-sh/uv:latest AS uvbin

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
ARG TORCH_VARIANT

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv ffmpeg libsndfile1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uvbin /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_PYTHON=python3.12 \
    UV_LINK_MODE=copy

# Dependency layer: cached until pyproject/lock change
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --extra ml --extra ${TORCH_VARIANT}

COPY src/ src/
RUN uv sync --frozen --no-dev --extra ml --extra ${TORCH_VARIANT}

ENV PATH=/app/.venv/bin:$PATH \
    HF_HOME=/models \
    TOLKA_MODEL_CACHE_DIR=/models \
    TOLKA_WORK_DIR=/data/work \
    TOLKA_DB_PATH=/data/tolka.sqlite3

RUN mkdir -p /models /data && chown -R 1000:1000 /models /data /app
USER 1000:1000

EXPOSE 8000
CMD ["uvicorn", "tolka.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
