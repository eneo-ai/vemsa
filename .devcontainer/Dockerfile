FROM mcr.microsoft.com/devcontainers/python:3.11-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends ripgrep ffmpeg libmagic1 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
