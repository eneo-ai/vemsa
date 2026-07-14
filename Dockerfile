# Whisper inference runs on a remote OpenAI-compatible endpoint; the only local ML is
# pyannote diarization. Torch pip wheels bundle their own CUDA/cuDNN libraries, so a
# plain Python base suffices — GPU use only needs the NVIDIA driver + container toolkit
# on the host.
ARG TORCH_VARIANT=gpu

FROM ghcr.io/astral-sh/uv:latest AS uvbin

FROM python:3.12-slim-bookworm
ARG TORCH_VARIANT

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uvbin /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_LINK_MODE=copy

# Dependency layer: cached until pyproject/lock change
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --extra diarize --extra ${TORCH_VARIANT}

COPY src/ src/
RUN uv sync --frozen --no-dev --extra diarize --extra ${TORCH_VARIANT}

ENV PATH=/app/.venv/bin:$PATH \
    HF_HOME=/models \
    TOLKA_MODEL_CACHE_DIR=/models \
    TOLKA_WORK_DIR=/data/work \
    TOLKA_DB_PATH=/data/tolka.sqlite3

RUN mkdir -p /models /data && chown -R 1000:1000 /models /data /app
USER 1000:1000

EXPOSE 8000
CMD ["uvicorn", "tolka.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
