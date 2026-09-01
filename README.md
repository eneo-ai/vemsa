# Vemsa

Vemsa (from Swedish *"vem sa?"* — *"who said?"*) is a standalone transcription and speaker-diarization service
built for Swedish (e.g. [KBLab/kb-whisper-large](https://huggingface.co/KBLab/kb-whisper-large)).
It offloads GPU-intensive whisper inference to an external OpenAI-compatible endpoint **when
possible**, while keeping the quality-critical stages — CTC forced alignment (word-precise
timestamps) and [pyannote](https://github.com/pyannote/pyannote-audio) speaker diarization —
local.

It exposes two front doors over one job engine:

- **Async job API** — submit, poll, cancel, and fetch through `/v1/jobs`
- **MCP facade** — streamable-HTTP MCP server at `/mcp` with `transcribe_audio`,
  `submit_transcription`, and `get_transcription` tools

FastMCP is only the adapter that implements the MCP protocol. It lets MCP-capable AI
assistants call Vemsa as tools; it does not transcribe audio or maintain a separate queue.
REST and MCP create and read the same jobs.

The service is platform-agnostic: any client that can speak HTTP (or MCP) can use it.

## Engine tiers

`VEMSA_ENGINE` selects how transcription runs (default `auto`):

| Engine | Whisper | Word timestamps | Needs extras | Use when |
| --- | --- | --- | --- | --- |
| `local` | in-process ([easytranscriber](https://github.com/kb-labb/easytranscriber)) | CTC forced alignment (best) | `local`, `diarize` | this box has a GPU, no external whisper |
| `hybrid` | remote endpoint | local CTC forced alignment via [easyaligner](https://github.com/kb-labb/easyaligner) of the remote transcript (WhisperX-style) | `align`, `diarize` | offload heavy whisper, keep precise timestamps — alignment needs only a wav2vec2-sized model |
| `remote` | remote endpoint | provider's own (quality varies by serving stack) | `diarize`, `align` | the whisper provider is contractually required to measure timestamps honestly |
| `diarize` | none — transcripts come from the caller | local forced alignment of the caller's transcript (or the caller's words with `VEMSA_DIARIZE_PREFER_ALIGN=false`) | `diarize`, `align` | the consumer transcribes with its own models; Vemsa adds only speaker labels |
| `fake` | none | canned | – | dev/smoke |

`auto` picks `hybrid` when `VEMSA_WHISPER_API_BASE` is set, otherwise `local`.

**Production posture: Vemsa runs on a GPU machine with the full align stack — that is a
requirement, not an option.** On the quality tiers (`local`, `hybrid`, `diarize`) whisper
contributes text only, wav2vec2 CTC forced alignment is the sole source of word timestamps,
and pyannote the sole source of speakers; no provider's decoder-heuristic timestamps ever
enter the result, and a job whose transcript cannot be aligned **fails loudly** instead of
degrading to a coarser rung. `remote` is the deliberate exception: it trusts the provider's
timestamps, which is only acceptable when the deployment puts quality requirements on that
provider (and can enforce them with `VEMSA_MIN_ALIGNMENT`). CPU-only operation is intended
for local testing with small files, expected to be slow.

In every tier, pyannote diarization runs locally and speakers are merged onto the timestamps
by maximal temporal overlap (word-level when words exist, whole-segment otherwise). The merge
is jitter-tolerant: a turn covering less than a quarter of a word's duration is not trusted
(the word inherits the previous speaker — but only across a short gap), and a mid-sentence
"island" of a couple of words whose neighbours on both sides agree on a different speaker is
relabelled to match them — a short untranscribed backchannel otherwise strands a word as its
own one-word segment (see `docs/ARCHITECTURE.md` → "Speaker attribution and segment shaping"
for the full rules; every threshold is tunable via `VEMSA_ATTR_*`). Labelled
words are grouped into output segments the way a human lines a transcript: a segment ends at
a speaker change, or at a pause (>1 s) that coincides with sentence-final punctuation — a
pause mid-sentence, or after a heading's colon, keeps the sentence together (a >15 s silence
splits regardless, so unpunctuated transcripts cannot collapse into one segment). Requests
may pass `num_speakers`, or `min_speakers`/`max_speakers`, as a clustering prior — pyannote
tends to over-split without one.

The remote endpoint is called with `response_format=verbose_json` and
`timestamp_granularities[]=word,segment`; anything OpenAI-compatible works (speaches,
faster-whisper-server, vLLM's audio API, ...).

## Word timestamps: trust and the alignment rungs

Diarization quality is word-timestamp quality: speakers are attached to whatever timeline
the words carry. Vemsa ranks its word sources and every result reports the rung that
produced it in the `alignment` field. The service is built for deployments that always
have a GPU and the align stack, so a tier whose rung is `forced` **fails the job loudly**
when alignment cannot run — it never silently degrades to a coarser rung. The lower rungs
exist only for deployments that deliberately trust an external timestamp source
(`VEMSA_ENGINE=remote`, or `VEMSA_DIARIZE_PREFER_ALIGN=false`) and can put quality
requirements on that provider.

| `alignment` | Word source | Precision |
| --- | --- | --- |
| `forced` | local CTC forced alignment of the transcript against the audio | word-precise (best) |
| `provider_words` | word timestamps from the whisper provider or the caller | word-precise **if** the provider measured honestly |
| `segment_split` | no words; segments spanning several speaker turns were split at the turn boundaries, slicing text by time proportion (snapped to punctuation/whitespace) | approximate — cuts can land a word or two off |
| `segment_only` | no words; each whole segment got its maximal-overlap speaker | speaker changes inside a segment are lost |

Two guards decide which rung a job lands on:

- **Plausibility gate.** Provider/caller word timelines are only trusted when they show a
  humanly possible speaking rate (≤ 4.5 words/s, pauses over 1 s excluded). Some serving
  stacks derive timestamps from decoder heuristics and emit compressed timelines (observed
  in the wild: ~7 words/s with a 19.5 s hole); merging speakers against such a timeline puts
  every turn in the wrong place, so implausible words are discarded and the transcript text
  is force-aligned against the audio instead.
- **Window trust.** Forced alignment normally uses the segments' start/end as anchors. When
  the words were rejected, the accompanying segment windows come from the same broken
  timeline (and segment windows can be implausibly compressed on their own), so the full
  transcript is aligned against the whole audio in one window instead.

Who wins when several sources are available differs by task, deliberately:

- **`task=transcribe`, hybrid tier**: forced alignment wins over provider words
  (WhisperX-style) — the provider's transcript text is kept, its timeline replaced.
- **`task=diarize`**: forced alignment wins here too (`VEMSA_DIARIZE_PREFER_ALIGN`, default
  on): caller-supplied words are set aside and the transcript is realigned against the audio —
  timestamps derived from the audio beat whatever the caller's provider decoded. Should
  alignment fail, the job fails; the set-aside words are never resurrected. Callers that
  measure their word timestamps honestly can restore caller-words-first with
  `VEMSA_DIARIZE_PREFER_ALIGN=false`.

The merge is logged (`alignment=forced words=<n>`), so the worker log always answers
"which timeline were the speakers attached to".

Each word may also carry a `probability` — a confidence score whose meaning follows the
rung: on `forced` it is the CTC forced-alignment score; on `provider_words` it is the ASR
decoder's posterior probability, passed through when the provider or caller reports one
(faster-whisper-derived servers do, plain OpenAI does not). It is `null` when the source
reports none, and values are not comparable across rungs. The `segment_split` and
`segment_only` rungs produce no words at all.

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
  -F file=@meeting.mp3 -F language=sv \
  -F 'vocabulary=["Anna Lindqvist","Çagri"]'
```

`vocabulary` is an optional list of names/terms likely to occur in the audio (meeting
participants, product names). It is passed to the whisper provider as the OpenAI-compatible
`prompt` field, biasing the decoder toward those spellings — a hint, not a guarantee, and in
ambiguous audio it can also bias *toward* hearing the hinted terms. Capped at 50 entries of
up to 64 characters (1024 total). Only accepted for `task=transcribe` (a diarize job runs no
ASR — 422); the local tier has no prompt support and logs-and-ignores it; a per-job
vocabulary overrides a static `prompt` configured via `VEMSA_WHISPER_EXTRA_FORM`.

Poll `GET /v1/jobs/{id}` until `status` is `completed`, then fetch `GET /v1/jobs/{id}/result`:

```json
{
  "job_id": "...",
  "status": "running",
  "stage": "diarizing",
  "queue_position": null,
  "created_at": "2026-08-26T12:00:00Z",
  "error": null
}
```

`status` is `queued`, `running`, `completed`, `failed`, or `cancelled`. `stage` is the
coarse, persisted processing stage: `queued`, `transcribing`, `aligning`, `diarizing`, or
`finalizing`. Queued jobs include their current global FIFO `queue_position`; it is `null`
after the worker claims the job.

Cancel a job with `DELETE /v1/jobs/{id}`. Cancellation is idempotent: an active job returns
`202` with `cancellation_requested=true`; a terminal job returns `200` unchanged; an unknown
or other-client job returns `404`. Queued jobs leave the queue immediately. Running work
stops at the next safe pipeline stage boundary, and the store rejects any completion that
races with cancellation.

```bash
curl -X DELETE http://localhost:8000/v1/jobs/$JOB_ID \
  -H "Authorization: Bearer $TOKEN"
```

Authenticated `GET /v1/health/ready` reports database and worker readiness alongside
`service_version`, `queue_accepting_jobs`, and `queued_jobs`; an unavailable database or
worker returns `503`.

```json
{
  "language": "sv",
  "duration_seconds": 3612.4,
  "model": "KBLab/kb-whisper-large",
  "alignment": "forced",
  "text": "[00:00:12 - 00:00:15] SPEAKER_00: ...",
  "segments": [
    {"start": 12.3, "end": 15.1, "speaker": "SPEAKER_00", "text": "...",
     "words": [{"word": "...", "start": 12.3, "end": 12.6, "probability": 0.97}]}
  ]
}
```

`speaker` is `null` when `diarize=false`. If `webhook_url` is given, the result is POSTed there
on completion. Results are retained for `VEMSA_RETENTION_HOURS` and then purged; source audio is
deleted as soon as the job reaches a terminal state.

### Diarize-only jobs (`task=diarize`)

When the transcript is produced elsewhere, Vemsa can add only the speaker labels: upload the
audio plus the transcript as JSON (`words`, `segments`, or both), and get back the same
`TranscriptionResult` shape with speakers merged in.

```bash
curl -X POST http://localhost:8000/v1/jobs \
  -H "Authorization: Bearer $TOKEN" \
  -F file=@meeting.mp3 -F task=diarize -F language=sv \
  -F 'words=[{"word":"hej","start":0.0,"end":0.4},{"word":"då","start":2.1,"end":2.3}]'
```

Timestamps are absolute seconds from the start of the uploaded audio. The recommended
payload is `segments` with **measured** windows — e.g. one segment per transcription chunk
whose boundaries the caller measured itself — because Vemsa force-aligns the text against
the audio for word-precise speaker changes (`VEMSA_DIARIZE_PREFER_ALIGN`, default on) and
trustworthy windows are the best alignment anchors; a single segment covering the whole
file also works. Caller-supplied `words` are used directly only when the deployment opts
into caller-words-first (`VEMSA_DIARIZE_PREFER_ALIGN=false`); whenever alignment is
needed, it is mandatory — a job whose transcript cannot be aligned fails rather than
degrading to a segment-level merge. The transcript is capped at
`VEMSA_MAX_TRANSCRIPT_BYTES` (default 8 MiB). `model` is echoed into the result and defaults
to `"external"` so the result never claims one of Vemsa's models transcribed. Speaker-count
priors (`num_speakers`, or `min_speakers`/`max_speakers`, all ≥ 1) are accepted on both
tasks, multipart or JSON.

Every engine tier accepts both tasks — one full deployment serves transcription and
diarize-only jobs side by side, and each request chooses. Optionally,
`VEMSA_ENGINE=diarize` locks a deployment down to diarize-only: no whisper endpoint needed,
and `task=transcribe` submissions are rejected with 422. The MCP tools remain
transcribe-only.

Auth is fail-closed static bearer authentication intended for internal server-to-server use.
`VEMSA_API_TOKENS` takes comma-separated `client_id=token` credentials in production, for
example `eneo=first-secret,automation=second-secret`. Jobs are owned by that client identity;
one client cannot read another client's status or transcript. The same credentials authorize
REST and MCP. Put internet-facing deployments behind an identity-aware gateway; interactive
MCP OAuth remains outside the v1 service boundary.

## Configuration

All settings via environment variables with the `VEMSA_` prefix (see `src/vemsa/config.py`):

| Variable | Default | Description |
| --- | --- | --- |
| `VEMSA_API_TOKENS` | *(required in production)* | Comma-separated `client_id=token` credentials |
| `VEMSA_ENVIRONMENT` | `development` | `development` / `test` / `production`; production requires named credentials and rejects the fake engine |
| `VEMSA_ENGINE` | `auto` | `auto` / `local` / `hybrid` / `remote` / `diarize` / `fake` (see Engine tiers) |
| `VEMSA_WHISPER_API_BASE` | – | OpenAI-compatible base URL, e.g. `http://whisper-host:8000/v1` (required for hybrid/remote) |
| `VEMSA_WHISPER_API_KEY` | – | Bearer token for the whisper endpoint |
| `VEMSA_WHISPER_TIMEOUT_S` | 3600 | Per-request timeout against the whisper endpoint |
| `VEMSA_WHISPER_EXTRA_FORM` | – | Extra `key=value` form fields sent with every provider transcription request; a job's `vocabulary` overrides a static `prompt` set here |
| `VEMSA_DEFAULT_MODEL` | `KBLab/kb-whisper-large` | Whisper model (name passed to the endpoint, or loaded locally) |
| `VEMSA_EMISSIONS_MODEL` | `KBLab/wav2vec2-large-voxrex-swedish` | Fallback CTC model for forced alignment when the job language has no `VEMSA_EMISSIONS_MODELS` entry |
| `VEMSA_EMISSIONS_MODELS` | `sv=KBLab/wav2vec2-large-voxrex-swedish` | Per-language CTC alignment models, comma-separated `lang=model`; an unmapped explicit language aligns with the fallback under a logged warning (`align.language_fallback`) |
| `VEMSA_DIARIZATION_MODEL` | `pyannote/speaker-diarization-community-1` | pyannote diarization pipeline (HF-gated; accept its license) |
| `VEMSA_DIARIZE_EXCLUSIVE` | `true` | Attribute words against the pipeline's exclusive diarization (one speaker at a time, matching what an ASR system would transcribe) instead of the raw overlapping turns |
| `VEMSA_DIARIZE_PREFER_ALIGN` | `true` | `task=diarize`: force-align even when the caller supplied words; a failed alignment fails the job. Set `false` only for callers whose word timestamps are honestly measured |
| `VEMSA_MIN_ALIGNMENT` | – | Quality floor (`forced` / `provider_words` / `segment_split` / `segment_only`): fail a job whose word-timestamp rung degrades below it instead of completing with a coarser result |
| `VEMSA_ATTR_*` | see `AttributionTuning` | Word→speaker attribution tuning (coverage floor, island smoothing limits, inheritance gap, segment splitting/merging thresholds) — one env knob per field of `AttributionTuning` in `pipeline/diarize.py`; defaults are the tested behaviour |
| `VEMSA_HF_TOKEN` | – | Hugging Face token (pyannote models are gated) |
| `VEMSA_MODEL_CACHE_DIR` | `./data/models` | Model cache (mount a volume) |
| `VEMSA_WORK_DIR` | `./data/work` | Temp audio storage |
| `VEMSA_DATABASE_URL` | – (required) | PostgreSQL URL for the job store |
| `VEMSA_MAX_AUDIO_BYTES` | 2 GiB | Upload/download size limit |
| `VEMSA_RETENTION_HOURS` | 72 | Result retention before purge |
| `VEMSA_PRELOAD_MODELS` | `false` | Load ML pipelines at startup instead of first job |
| `VEMSA_ALLOW_PRIVATE_URLS` | `false` | Allow `source_url` to resolve to private networks |
| `VEMSA_SOURCE_ALLOWED_HOSTS` | – | Optional comma-separated source hostname allowlist |
| `VEMSA_WEBHOOK_ALLOWED_HOSTS` | – | Optional comma-separated webhook hostname allowlist |
| `VEMSA_WEBHOOK_SIGNING_SECRET` | – | HMAC-SHA256 secret for signed webhook delivery |
| `VEMSA_MAX_QUEUED_JOBS` | `100` | Global active-job admission limit |
| `VEMSA_MAX_QUEUED_JOBS_PER_CLIENT` | `10` | Per-client active-job admission limit |
| `VEMSA_RUN_WORKER` | `true` | Run a worker in this process; production separates API and worker |
| `VEMSA_LOG_FORMAT` | `json` | Structured `json` or human-readable `text` logs |

`HF_TOKEN` is also honored for the Hugging Face hub. You must accept the pyannote model license
on Hugging Face (`pyannote/speaker-diarization-community-1`) for the diarization stage to
download it.

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
VEMSA_API_TOKENS=dev VEMSA_ENGINE=fake uv run uvicorn vemsa.main:create_app --factory
```

A devcontainer is included (`.devcontainer/`): reopen in container to get Python 3.11, ffmpeg,
uv, and the full CPU ML stack pre-installed.

## Docker

```bash
# Production topology: PostgreSQL + API + GPU worker
export POSTGRES_PASSWORD='replace-with-a-random-secret'
export VEMSA_API_TOKENS='operator=replace-with-a-random-token'
docker compose up --build

docker build --build-arg TORCH_VARIANT=cpu -t vemsa:cpu .        # CPU-only ML
docker build --build-arg ML_EXTRAS="diarize align" -t vemsa .    # hybrid/remote only (smaller)
```

Prebuilt images are published to GHCR on every `main` push and release tag — `latest` /
`vX.Y.Z` with CUDA torch and a `-cpu` twin of every tag (both amd64-only; torchcodec has
no linux/arm64 wheels, so Apple-silicon development runs natively via `uv sync`):

```bash
docker pull ghcr.io/eneo-ai/vemsa:latest         # NVIDIA GPU hosts
docker pull ghcr.io/eneo-ai/vemsa:latest-cpu     # CPU-only hosts
VEMSA_IMAGE=ghcr.io/eneo-ai/vemsa:latest docker compose up --no-build
```

See `docs/PRODUCTION.md` (“Container images”) for choosing GPU vs CPU, host requirements,
and digest pinning.

Compose runs the API and GPU worker separately. PostgreSQL owns jobs, leases, worker
heartbeats, and the durable webhook outbox; the containers share `/data` for temporary audio
on one Docker host. Set `VEMSA_PORT` if host port 8000 is already occupied. Multi-node workers
will require object storage instead of this shared local volume.

Operational endpoints:

- `GET /livez` — process liveness
- `GET /readyz` (and compatibility alias `/healthz`) — database and worker readiness
- `GET /v1/health/ready` — authenticated service readiness and queue admission state
- `GET /metrics` — Prometheus metrics; requires the same bearer authentication

The worker logs which job store it opened at startup, a heartbeat with queued/running
counts every 30s, and per-job progress (stage, elapsed time) while processing — a silent
worker is stopped or stuck, never just idle.

See [`docs/PRODUCTION.md`](docs/PRODUCTION.md) for security, webhooks, backups, and rollout,
and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the internal architecture and the
responsibility split between Vemsa and its consumers (eneo).

## Status

- ✅ Job engine, REST API, MCP facade, engine selection, remote whisper client, speaker-merge
  (word- and segment-level, proportional splitting), plausibility gating, renderer — tested
  without the torch stack
- ✅ PostgreSQL job leasing, per-client ownership, queue admission limits, durable signed
  webhooks, structured logs, metrics, and split API/worker deployment
- ✅ Forced alignment (easyaligner: silero VAD + wav2vec2 CTC) and pyannote diarization with
  speaker bounds, verified end to end on CPU in the devcontainer — a whole-file Swedish
  segment came back word-precise with correct alternating speakers (`alignment: "forced"`)
- ⏳ `local` engine (easytranscriber whisper) verification on a GPU box: Swedish sample
  end-to-end, auto-detect language
- ⏳ CUDA runs of alignment and diarization (verified on CPU only); non-Swedish alignment
  quality — the single emissions model is Swedish, so `en` jobs align against a Swedish
  acoustic model

## Acknowledgements

Vemsa builds on the excellent work of [KBLab](https://kb-labb.github.io/) at the National
Library of Sweden:

- [easytranscriber](https://github.com/kb-labb/easytranscriber) — the transcription toolkit
  that powers the `local` engine and inspired Vemsa's pipeline design
- [easyaligner](https://github.com/kb-labb/easyaligner) — CTC forced alignment used for
  word-precise timestamps in the `hybrid` engine and for `task=diarize` transcripts
- [kb-whisper](https://huggingface.co/KBLab/kb-whisper-large) and
  [wav2vec2-large-voxrex-swedish](https://huggingface.co/KBLab/wav2vec2-large-voxrex-swedish) —
  the default Swedish speech models

Speaker diarization is provided by [pyannote-audio](https://github.com/pyannote/pyannote-audio).

## License

AGPL-3.0-or-later, © Sundsvalls Kommun.
