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

FastMCP is only the adapter that implements the MCP protocol. It lets MCP-capable AI
assistants call Tolka as tools; it does not transcribe audio or maintain a separate queue.
REST and MCP create and read the same jobs.

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

Auth is fail-closed static bearer authentication intended for internal server-to-server use.
`TOLKA_API_TOKENS` takes comma-separated `client_id=token` credentials in production, for
example `eneo=first-secret,automation=second-secret`. Jobs are owned by that client identity;
one client cannot read another client's status or transcript. The same credentials authorize
REST and MCP. Put internet-facing deployments behind an identity-aware gateway; interactive
MCP OAuth remains outside the v1 service boundary.

## Configuration

All settings via environment variables with the `TOLKA_` prefix (see `src/tolka/config.py`):

| Variable | Default | Description |
| --- | --- | --- |
| `TOLKA_API_TOKENS` | *(required in production)* | Comma-separated `client_id=token` credentials |
| `TOLKA_ENVIRONMENT` | `development` | `development` / `test` / `production`; production requires named credentials and rejects the fake engine |
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
| `TOLKA_DATABASE_URL` | – | PostgreSQL URL; when set, PostgreSQL replaces SQLite |
| `TOLKA_MAX_AUDIO_BYTES` | 2 GiB | Upload/download size limit |
| `TOLKA_RETENTION_HOURS` | 72 | Result retention before purge |
| `TOLKA_PRELOAD_MODELS` | `false` | Load ML pipelines at startup instead of first job |
| `TOLKA_ALLOW_PRIVATE_URLS` | `false` | Allow `source_url` to resolve to private networks |
| `TOLKA_SOURCE_ALLOWED_HOSTS` | – | Optional comma-separated source hostname allowlist |
| `TOLKA_WEBHOOK_ALLOWED_HOSTS` | – | Optional comma-separated webhook hostname allowlist |
| `TOLKA_WEBHOOK_SIGNING_SECRET` | – | HMAC-SHA256 secret for signed webhook delivery |
| `TOLKA_MAX_QUEUED_JOBS` | `100` | Global active-job admission limit |
| `TOLKA_MAX_QUEUED_JOBS_PER_CLIENT` | `10` | Per-client active-job admission limit |
| `TOLKA_RUN_WORKER` | `true` | Run a worker in this process; production separates API and worker |
| `TOLKA_LOG_FORMAT` | `json` | Structured `json` or human-readable `text` logs |

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
# Production topology: PostgreSQL + API + GPU worker
export POSTGRES_PASSWORD='replace-with-a-random-secret'
export TOLKA_API_TOKENS='operator=replace-with-a-random-token'
docker compose up --build

docker build --build-arg TORCH_VARIANT=cpu -t tolka:cpu .        # CPU-only ML
docker build --build-arg ML_EXTRAS="diarize align" -t tolka .    # hybrid/remote only (smaller)
```

Prebuilt images are published to GHCR on every `main` push and release tag — `latest` /
`vX.Y.Z` with CUDA torch (amd64) and a `-cpu` twin of every tag (amd64 + arm64):

```bash
docker pull ghcr.io/eneo-ai/tolka:latest         # NVIDIA GPU hosts
docker pull ghcr.io/eneo-ai/tolka:latest-cpu     # CPU-only hosts (incl. Apple silicon)
TOLKA_IMAGE=ghcr.io/eneo-ai/tolka:latest docker compose up --no-build
```

See `docs/PRODUCTION.md` (“Container images”) for choosing GPU vs CPU, host requirements,
and digest pinning.

Compose runs the API and GPU worker separately. PostgreSQL owns jobs, leases, worker
heartbeats, and the durable webhook outbox; the containers share `/data` for temporary audio
on one Docker host. Set `TOLKA_PORT` if host port 8000 is already occupied. Multi-node workers
will require object storage instead of this shared local volume.

Operational endpoints:

- `GET /livez` — process liveness
- `GET /readyz` (and compatibility alias `/healthz`) — database and worker readiness
- `GET /metrics` — Prometheus metrics; requires the same bearer authentication

See [`docs/PRODUCTION.md`](docs/PRODUCTION.md) for security, webhooks, backups, and rollout.

## Status

- ✅ Job engine, REST API, MCP facade, engine selection, remote whisper client, speaker-merge
  (word- and segment-level), renderer — tested without the torch stack
- ✅ PostgreSQL job leasing, per-client ownership, queue admission limits, durable signed
  webhooks, structured logs, metrics, and split API/worker deployment
- ⏳ `local` engine (easytranscriber) verification on a GPU box: Swedish sample end-to-end,
  auto-detect language, Swedish alignment tokenizer
- ⏳ `hybrid` engine: easyaligner invocation is best-effort against the documented API and
  guarded by a runtime fallback — validate against a real installation
- ⏳ pyannote diarization verification (HF gating, `GPU-VERIFY` markers in code)

## Acknowledgements

Tolka builds on the excellent work of [KBLab](https://kb-labb.github.io/) at the National
Library of Sweden:

- [easytranscriber](https://github.com/kb-labb/easytranscriber) — the transcription toolkit
  that powers the `local` engine and inspired Tolka's pipeline design
- [easyaligner](https://github.com/kb-labb/easyaligner) — CTC forced alignment used for
  word-precise timestamps in the `hybrid` engine
- [kb-whisper](https://huggingface.co/KBLab/kb-whisper-large) and
  [wav2vec2-large-voxrex-swedish](https://huggingface.co/KBLab/wav2vec2-large-voxrex-swedish) —
  the default Swedish speech models

Speaker diarization is provided by [pyannote-audio](https://github.com/pyannote/pyannote-audio).

## License

AGPL-3.0-or-later, © Sundsvalls Kommun.
