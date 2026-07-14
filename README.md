# Tolka

Tolka (Swedish: *to interpret*) is a standalone transcription and speaker-diarization service
built for Swedish (e.g. [KBLab/kb-whisper-large](https://huggingface.co/KBLab/kb-whisper-large)).
It offloads GPU-intensive whisper inference to an external OpenAI-compatible endpoint **when
possible**, while keeping the quality-critical stages — CTC forced alignment (word-precise
timestamps) and [pyannote](https://github.com/pyannote/pyannote-audio) speaker diarization —
local.

It exposes two front doors over one job engine:

- **Async job API** — `POST /v1/jobs`, `GET /v1/jobs/{id}`, `GET /v1/jobs/{id}/result`
- **MCP facade** — streamable-HTTP MCP server at `/mcp` with `transcribe_audio`,
  `submit_transcription`, and `get_transcription` tools

The service is platform-agnostic: any client that can speak HTTP (or MCP) can use it.

## Engine tiers

`TOLKA_ENGINE` selects how transcription runs (default `auto`):

| Engine | Whisper | Word timestamps | Needs extras | Use when |
| --- | --- | --- | --- | --- |
| `local` | in-process ([easytranscriber](https://github.com/kb-labb/easytranscriber)) | CTC forced alignment (best) | `local`, `diarize` | this box has a GPU, no external whisper |
| `hybrid` | remote endpoint | local CTC forced alignment via [easyaligner](https://github.com/kb-labb/easyaligner) of the remote transcript (WhisperX-style) | `align`, `diarize` | offload heavy whisper, keep precise timestamps — alignment needs only a wav2vec2-sized model |
| `remote` | remote endpoint | provider's own (quality varies by serving stack) | `diarize` | lightest footprint, provider timestamps good enough |
| `fake` | none | canned | – | dev/smoke |

`auto` picks `hybrid` when `TOLKA_WHISPER_API_BASE` is set, otherwise `local`. The hybrid
engine also degrades at runtime: if the alignment stack is missing or alignment fails, it
falls back to the provider's timestamps rather than failing the job.

In every tier, pyannote diarization runs locally and speakers are merged onto the timestamps
by maximal temporal overlap (word-level when words exist, whole-segment otherwise).

The remote endpoint is called with `response_format=verbose_json` and
`timestamp_granularities[]=word,segment`; anything OpenAI-compatible works (speaches,
faster-whisper-server, vLLM's audio API, ...).

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
| `TOLKA_ENGINE` | `auto` | `auto` / `local` / `hybrid` / `remote` / `fake` (see Engine tiers) |
| `TOLKA_WHISPER_API_BASE` | – | OpenAI-compatible base URL, e.g. `http://whisper-host:8000/v1` (required for hybrid/remote) |
| `TOLKA_WHISPER_API_KEY` | – | Bearer token for the whisper endpoint |
| `TOLKA_WHISPER_TIMEOUT_S` | 3600 | Per-request timeout against the whisper endpoint |
| `TOLKA_DEFAULT_MODEL` | `KBLab/kb-whisper-large` | Whisper model (name passed to the endpoint, or loaded locally) |
| `TOLKA_EMISSIONS_MODEL` | `KBLab/wav2vec2-large-voxrex-swedish` | CTC model for forced alignment (local/hybrid) |
| `TOLKA_HF_TOKEN` | – | Hugging Face token (pyannote models are gated) |
| `TOLKA_MODEL_CACHE_DIR` | `./data/models` | Model cache (mount a volume) |
| `TOLKA_WORK_DIR` | `./data/work` | Temp audio storage |
| `TOLKA_DB_PATH` | `./data/tolka.sqlite3` | SQLite job store |
| `TOLKA_MAX_AUDIO_BYTES` | 2 GiB | Upload/download size limit |
| `TOLKA_RETENTION_HOURS` | 72 | Result retention before purge |
| `TOLKA_PRELOAD_MODELS` | `false` | Load ML pipelines at startup instead of first job |
| `TOLKA_ALLOW_PRIVATE_URLS` | `false` | Allow `source_url` to resolve to private networks |

`HF_TOKEN` is also honored for the Hugging Face hub. You must accept the pyannote model licenses
on Hugging Face (`pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`) for the
diarization stage to download them.

## Development

Requires [uv](https://docs.astral.sh/uv/). The ML stacks are behind extras so unit tests run
without any of them:

```bash
uv sync --group dev                               # all tests, no torch
uv run pytest
uv run ruff format --check . && uv run ruff check .

# ML extras (each tier's needs are listed under Engine tiers):
#   diarize = pyannote, align = easyaligner, local = easytranscriber
uv sync --group dev --extra diarize --extra align --extra local --extra cpu   # everything, CPU
uv sync --frozen --no-dev --extra diarize --extra align --extra gpu           # hybrid prod, CUDA 12.8
```

Always pair ML extras with exactly one of `--extra cpu` / `--extra gpu` — those route torch to
the right package index and are mutually exclusive.

Run a local server against the fake engine (no ML stack, no whisper endpoint needed):

```bash
TOLKA_API_TOKENS=dev TOLKA_ENGINE=fake uv run uvicorn tolka.main:create_app --factory
```

A devcontainer is included (`.devcontainer/`): reopen in container to get Python 3.11, ffmpeg,
uv, and the full CPU ML stack pre-installed.

## Docker

```bash
docker compose up --build                                        # all engine tiers, GPU
docker build --build-arg TORCH_VARIANT=cpu -t tolka:cpu .        # CPU-only ML
docker build --build-arg ML_EXTRAS="diarize align" -t tolka .    # hybrid/remote only (smaller)
```

The image is plain `python:3.12-slim` — torch's pip wheels bundle their CUDA libraries, so GPU
use only needs the NVIDIA driver and container toolkit on the host. The compose file mounts
named volumes for the model cache (`/models`) and job data (`/data`), and reserves one GPU.
Set `TOLKA_API_TOKENS` (and `TOLKA_WHISPER_API_BASE` + `HF_TOKEN` as applicable) in the
environment.

## Status

- ✅ Job engine, REST API, MCP facade, engine selection, remote whisper client, speaker-merge
  (word- and segment-level), renderer — tested without the torch stack
- ⏳ `local` engine (easytranscriber) verification on a GPU box: Swedish sample end-to-end,
  auto-detect language, Swedish alignment tokenizer
- ⏳ `hybrid` engine: easyaligner invocation is best-effort against the documented API and
  guarded by a runtime fallback — validate against a real installation
- ⏳ pyannote diarization verification (HF gating, `GPU-VERIFY` markers in code)

## License

AGPL-3.0-or-later, © Sundsvalls Kommun.
