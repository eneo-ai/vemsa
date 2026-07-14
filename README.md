# Tolka

Tolka (Swedish: *to interpret*) is a standalone transcription and speaker-diarization service.
It wraps [kb-labb/easytranscriber](https://github.com/kb-labb/easytranscriber) for high-quality
Swedish speech-to-text with word-level timestamps (VAD → Whisper → forced alignment), and adds
speaker diarization via [pyannote](https://github.com/pyannote/pyannote-audio) on top.

It exposes two front doors over one job engine:

- **Async job API** — `POST /v1/jobs`, `GET /v1/jobs/{id}`, `GET /v1/jobs/{id}/result`
- **MCP facade** — streamable-HTTP MCP server at `/mcp` with `transcribe_audio`,
  `submit_transcription`, and `get_transcription` tools

The service is platform-agnostic: any client that can speak HTTP (or MCP) can use it.

## Job API

Submit a job with a source URL (JSON) or a direct upload (multipart):

```bash
curl -X POST http://localhost:8000/v1/jobs \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"source_url": "https://example.org/meeting.mp3", "language": "sv", "diarize": true}'
# -> 202 {"job_id": "...", "status": "queued"}

curl -X POST http://localhost:8000/v1/jobs \
  -H "Authorization: Bearer $TOKEN" \
  -F file=@meeting.mp3 -F language=sv
```

Poll `GET /v1/jobs/{id}` until `status` is `completed`, then fetch `GET /v1/jobs/{id}/result`:

```json
{
  "language": "sv",
  "duration_seconds": 3612.4,
  "model": "KBLab/kb-whisper-large",
  "text": "[00:00:12 - 00:00:15] SPEAKER_00: ...",
  "segments": [
    {"start": 12.3, "end": 15.1, "speaker": "SPEAKER_00", "text": "...",
     "words": [{"word": "...", "start": 12.3, "end": 12.6}]}
  ]
}
```

`speaker` is `null` when `diarize=false`. If `webhook_url` is given, the result is POSTed there
on completion. Results are retained for `TOLKA_RETENTION_HOURS` and then purged; source audio is
deleted as soon as the job finishes.

Auth is a static bearer token. `TOLKA_API_TOKENS` takes a comma-separated list so tokens can be
rotated without downtime. The same tokens authorize the MCP endpoint. (Static bearer suits
server-to-server use; public MCP clients increasingly expect OAuth — out of scope for v1.)

## Configuration

All settings via environment variables with the `TOLKA_` prefix (see `src/tolka/config.py`):

| Variable | Default | Description |
| --- | --- | --- |
| `TOLKA_API_TOKENS` | *(required)* | Comma-separated bearer tokens |
| `TOLKA_HF_TOKEN` | – | Hugging Face token (pyannote models are gated) |
| `TOLKA_DEFAULT_MODEL` | `KBLab/kb-whisper-large` | Whisper model |
| `TOLKA_MODEL_CACHE_DIR` | `./data/models` | Model cache (mount a volume) |
| `TOLKA_WORK_DIR` | `./data/work` | Temp audio storage |
| `TOLKA_DB_PATH` | `./data/tolka.sqlite3` | SQLite job store |
| `TOLKA_MAX_AUDIO_BYTES` | 2 GiB | Upload/download size limit |
| `TOLKA_RETENTION_HOURS` | 72 | Result retention before purge |
| `TOLKA_PRELOAD_MODELS` | `false` | Load models at startup instead of first job |
| `TOLKA_ALLOW_PRIVATE_URLS` | `false` | Allow `source_url` to resolve to private networks |
| `TOLKA_FAKE_ENGINE` | `false` | Use a canned-output engine (dev/smoke only) |

`HF_TOKEN` is also honored for the Hugging Face hub. You must accept the pyannote model licenses
on Hugging Face (`pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`) for the
diarization stage to download them.

## Development

Requires [uv](https://docs.astral.sh/uv/). The ML stack (torch, easytranscriber, pyannote) is
behind extras so unit tests run without it:

```bash
uv sync --group dev                          # API + engine tests, no torch
uv run pytest
uv run ruff format --check . && uv run ruff check .

uv sync --group dev --extra ml --extra cpu   # full stack, CPU wheels (dev/devcontainer)
uv sync --frozen --no-dev --extra ml --extra gpu  # prod, CUDA 12.8 wheels
```

Always pair `--extra ml` with exactly one of `--extra cpu` / `--extra gpu` — the extras route
torch to the right package index and are mutually exclusive.

Run a local server against the fake engine (no ML stack needed):

```bash
TOLKA_API_TOKENS=dev TOLKA_FAKE_ENGINE=1 uv run uvicorn tolka.main:create_app --factory
```

A devcontainer is included (`.devcontainer/`): reopen in container to get Python 3.11, ffmpeg,
uv, and the CPU ML stack pre-installed.

## Docker

```bash
docker compose up --build     # GPU (requires nvidia container toolkit)
docker build --build-arg TORCH_VARIANT=cpu -t tolka:cpu .   # CPU image
```

The compose file mounts named volumes for the model cache (`/models`) and job data (`/data`),
and reserves one NVIDIA GPU. Set `TOLKA_API_TOKENS` and `HF_TOKEN` in the environment.

## Status

- ✅ Job engine, REST API, MCP facade, speaker-merge, renderer — tested on CPU without GPU
- ⏳ GPU verification of the real easytranscriber pipeline (Swedish sample end-to-end)
- ⏳ Swedish forced-alignment tokenizer/emissions model selection
  (`KBLab/wav2vec2-large-voxrex-swedish` is the candidate)

Search the code for `GPU-VERIFY` to find the spots that need validation on a GPU box.

## License

AGPL-3.0-or-later, © Sundsvalls Kommun. Wraps MIT-licensed
[easytranscriber](https://github.com/kb-labb/easytranscriber) (KBLab).
