# Architecture

This document describes how Vemsa is put together internally and where the boundary runs
between Vemsa and its primary consumer, [eneo](https://github.com/eneo-ai/eneo). The
README covers the *what* (API, engine tiers, configuration); this covers the *how* and
the *why of the split*.

## System overview

Vemsa is a single-purpose service: audio in, speaker-labelled transcript out, asynchronously.
It is deliberately not a platform — it has no tenants, no users, no UI, no long-term storage.
Consumers integrate over plain HTTP (or MCP) with a static service credential.

```mermaid
flowchart LR
    subgraph consumer["Consumer (e.g. eneo)"]
        flow[flow execution worker]
    end
    subgraph vemsa["Vemsa deployment"]
        api["api process<br/>REST /v1/jobs + MCP /mcp"]
        pg[("PostgreSQL<br/>jobs · leases · outbox")]
        worker["worker process<br/>ML pipeline (GPU)"]
        vol[("/data volume<br/>temp audio")]
    end
    whisper["external whisper endpoint<br/>(hybrid/remote tiers only)"]

    flow -- "submit / poll / cancel" --> api
    api --> pg
    api -- upload --> vol
    worker -- "lease (SKIP LOCKED)" --> pg
    worker -- read/delete --> vol
    worker -. optional .-> whisper
    worker -- "signed webhook" -.-> flow
```

## Process topology

One image, three services (see `compose.yaml`):

| Service | Role | Loads ML models |
| --- | --- | --- |
| `api` | REST + MCP front door: admission, auth, status, results, cancellation | never |
| `worker` | claims jobs, runs the pipeline, delivers webhooks, purges expired results | yes (owns the GPU) |
| `postgres` | the only source of truth: jobs, results, leases, worker heartbeats, webhook outbox | – |

The API stays responsive regardless of pipeline load because it never touches torch. The two
processes share nothing but PostgreSQL and the `/data` volume (temporary audio), which is what
limits the supplied Compose topology to one Docker host — multi-node workers need object
storage first. `VEMSA_RUN_WORKER=true` collapses both into one process for development.

## Module map

```
src/vemsa/
├── main.py             FastAPI app factory; mounts REST, MCP, health, metrics
├── config.py           all VEMSA_* settings (pydantic-settings)
├── worker.py           worker entrypoint: store + engine + JobQueue
├── security.py         outbound URL policy (schemes, private nets, allowlists)
├── observability.py    structured JSON logging, request/job contextvars, Prometheus metrics
├── api/
│   ├── jobs.py         POST/GET/DELETE /v1/jobs — admission caps, multipart & JSON
│   ├── auth.py         static bearer tokens → client_id; job ownership
│   └── health.py       /livez, /readyz, /v1/health/ready
├── mcp/server.py       FastMCP facade — protocol adapter over the same job store
├── jobs/
│   ├── models.py       Job, TranscriptionResult, Segment, Word — the wire contract
│   ├── store.py        JobStore interface + lifecycle state machine
│   ├── postgres_store.py  SKIP LOCKED leasing, heartbeats, outbox, retention
│   └── queue.py        worker loop: claim → run engine in thread → finalize;
│                       lease-renewal thread, webhook delivery, purge loop
└── pipeline/
    ├── factory.py      VEMSA_ENGINE → engine implementation
    ├── base.py         TranscriptionEngine protocol (blocking, stage-reporting)
    ├── fetch.py        source_url streaming download with size/redirect policy
    ├── transcribe.py   local tier: easytranscriber (whisper + VAD + CTC alignment)
    ├── whisper_api.py  remote tier + the OpenAI-compatible provider client
    ├── hybrid.py       hybrid tier: remote whisper text + local forced alignment
    ├── align.py        easyaligner wrapper: wav2vec2 CTC forced alignment
    ├── diarize.py      pyannote Diarizer + pure word↔speaker assignment logic
    ├── label.py        task=diarize: label an externally produced transcript
    ├── diarize_only.py diarize tier: no ASR constructed at all
    ├── render.py       [HH:MM:SS - HH:MM:SS] SPEAKER_00: line rendering
    └── fake.py         canned engine for dev/tests
```

The split inside `pipeline/` mirrors the quality doctrine: everything that decides *where
words land in time and who said them* (`align.py`, `diarize.py`, `label.py`) is local code
under Vemsa's control, while ASR text production is pluggable (in-process, remote endpoint,
or skipped entirely for `task=diarize`).

## Job lifecycle

1. **Admission** (`api/jobs.py`): authenticate → validate the request against the deployment
   tier (e.g. `task=transcribe` on `VEMSA_ENGINE=diarize` → 422) → enforce
   `VEMSA_MAX_QUEUED_JOBS` / `..._PER_CLIENT` (429, pre-body) → persist the job `queued`
   and stream the upload to `/data`. The response is an immediate `202` with `job_id`.
2. **Claim** (`jobs/queue.py`): a worker selects the oldest queued job with
   `FOR UPDATE SKIP LOCKED` and takes an expiring lease. Leases plus worker heartbeats are
   what makes crash recovery automatic: a dead worker's lease lapses and the job is
   re-leased, without an attempt cap — the consumer's deadline is the backstop.
3. **Pipeline**: the engine runs synchronously in a worker thread, reporting coarse stages
   (`transcribing → aligning → diarizing → finalizing`) that are persisted and visible to
   pollers. Lease renewal and heartbeats run on a dedicated thread with its own database
   connection, so GIL-heavy ML stages cannot let the lease lapse mid-job.
4. **Terminal**: result rows, webhook outbox events, and status flip in one transaction.
   Cancellation (`DELETE`) is idempotent and race-safe — the store refuses a completion
   that lost the race. Source audio is deleted at any terminal state; results are purged
   after `VEMSA_RETENTION_HOURS` by the worker's purge loop.

## Pipeline data flow

All tiers converge on the same back half — that is the design's core invariant:

```
                      task=transcribe                        task=diarize
        ┌──────────────┬────────────────┬──────────┐    ┌──────────────────┐
        │ local        │ hybrid         │ remote   │    │ caller transcript │
        │ easytranscr. │ remote whisper │ remote   │    │ (words/segments)  │
        │ (in-process) │ text only      │ whisper  │    └────────┬─────────┘
        └──────┬───────┴──────┬─────────┴────┬─────┘             │
               │              ▼              │                   ▼
               │   forced alignment          │        forced alignment of the
               │   (easyaligner CTC)         │        caller's text (default)
               ▼              │              ▼                   │
        ┌──────┴──────────────┴──────────────┴───────────────────┴─────┐
        │ word-timestamp trust: forced > provider_words >              │
        │ segment_split > segment_only  (plausibility-gated, README)   │
        ├──────────────────────────────────────────────────────────────┤
        │ pyannote diarization (always local)                          │
        │ word↔speaker attribution + island smoothing                  │
        │ segment grouping at sentence-final pauses                    │
        │ render_text → TranscriptionResult                            │
        └──────────────────────────────────────────────────────────────┘
```

Only the *text source* varies by tier. Word timestamps are derived from the audio by local
CTC forced alignment (the `forced` rung) — mandatory on the quality tiers, where a failed
or unavailable alignment fails the job loudly rather than degrading to provider timestamps
or a segment-level merge (`remote`, and `VEMSA_DIARIZE_PREFER_ALIGN=false`, are the
deliberate exceptions that trust an external timestamp source). pyannote always runs
locally, and every tier — including `task=diarize` — renders through the same
attribution/grouping/render code, so consumers get one output shape regardless of who
transcribed.

### Speaker attribution and segment shaping

The merge from diarization turns onto the word timeline (`pipeline/diarize.py`, pure and
CI-tested without torch) applies, in order:

- **Exclusive diarization** (`VEMSA_DIARIZE_EXCLUSIVE`, default on): words are attributed
  against pyannote's one-speaker-at-a-time track, matching what an ASR system would have
  transcribed during overlap.
- **Maximal temporal overlap** with a coverage floor: a word takes the turn covering most
  of it, but a turn overlapping less than a quarter of the word's duration is not trusted —
  the word inherits the previous speaker (only across a gap of at most 2 s; after a longer
  silence the nearest turn wins) instead of jumping on alignment jitter.
- **Island smoothing**: a run of at most two words within a 3 s span (or any run under one
  second) whose neighbours on both sides agree on a different speaker is relabelled to that
  speaker — a short untranscribed backchannel otherwise flips the exclusive track and
  strands a mid-sentence word as its own one-word segment. Islands starting right after
  sentence-final punctuation are kept, since those are plausibly real one-word turns —
  unless the island reads as glued into the following words mid-sentence (no punctuation of
  its own, next word continues lowercase), which is a sentence's first word caught in a
  backchannel window, not an interjection.
- **Boundary snapping**: a speaker change sitting mid-sentence (no sentence-final
  punctuation before it, lowercase continuation after it) is moved to the nearest sentence
  boundary within a few words — diarization turns routinely start a beat late or early
  relative to the aligned words, clipping a turn's first words onto the previous speaker
  ("...vara här. Jag | har sett..."). An uppercase start (a genuine interruption), a long
  silence at the change, or no sentence end within reach leaves the change untouched.
- **Segment grouping**: labelled words become output segments the way a human lines a
  transcript — break at speaker changes, and at pauses over 1 s only when they coincide
  with sentence-final punctuation (a >15 s silence breaks regardless).
- **Orphan merging**: a one-word segment whose speaker appears on neither side (smoothing
  needs agreeing neighbours, so three-speaker jitter and the transcript's edges get past
  it) is merged into the neighbour it forms a sentence with; without grammatical glue it
  is left alone.

Every threshold above is env-tunable via the `VEMSA_ATTR_*` settings (see
`AttributionTuning` in `pipeline/diarize.py`); the defaults are the tested behaviour.

## Eneo and Vemsa: the responsibility split

Eneo is a multi-tenant AI platform; Vemsa is its transcription/diarization engine. The
boundary is drawn so that each side owns what only it can know:

| Concern | Owner | Notes |
| --- | --- | --- |
| End-user identity, tenants, permissions | **eneo** | Vemsa sees exactly one credential (`client_id=eneo`) for all of eneo's tenants |
| UI, flows, orchestration, retries, deadlines | **eneo** | Vemsa re-leases crashed jobs forever; eneo's poll deadline is the backstop |
| Original audio custody & long-term storage | **eneo** | Vemsa keeps audio only while a job is active, results only for `VEMSA_RETENTION_HOURS` |
| ASR in eneo's own transcription flow | **eneo** | eneo may transcribe with its own models and use Vemsa for labels only (`task=diarize`) |
| Chunk boundaries in that flow | **eneo** | eneo sends one segment per chunk with windows *it measured itself* — the only trustworthy timestamps in that path |
| Job queueing, admission, leasing, crash recovery | **Vemsa** | PostgreSQL leases; per-client caps bound eneo's total fan-in across tenants |
| Whisper ASR when asked to transcribe | **Vemsa** | in-process on GPU (`local`) or brokered to an OpenAI-compatible endpoint (`hybrid`/`remote`) |
| Word timestamps | **Vemsa** | audio-derived CTC forced alignment wins over anything a provider or caller decoded |
| Speaker diarization & attribution | **Vemsa** | pyannote always local; attribution/smoothing/grouping identical for both tasks |
| Rendered output contract | **Vemsa** | `TranscriptionResult` + the `[HH:MM:SS - HH:MM:SS] SPEAKER_00:` text format |

Two integration paths use the same deployment, chosen per request:

1. **Vemsa transcribes** — eneo uploads audio (`task=transcribe`), Vemsa runs the full
   pipeline and returns text with speakers.
2. **Eneo transcribes, Vemsa labels** — eneo runs its own ASR in chunks, then submits the
   audio plus the chunk texts as measured segment windows (`task=diarize`). Vemsa
   force-aligns the text inside those windows and merges speakers; eneo's provider word
   timestamps are neither needed nor wanted in the payload, because decoder-heuristic
   timelines are exactly what the plausibility gate exists to reject.

In both paths eneo passes `result.text` verbatim into its flow output — Vemsa owns the
transcript's final shape. The wire contract (multipart part names, status/stage enums,
cancellation semantics, `TranscriptionResult`, the rendered line format) is
change-controlled and must move in lockstep with eneo's `RemoteTranscriptionClient`
contract tests; see "Consumer integrations (eneo)" in [PRODUCTION.md](PRODUCTION.md) for
the deployment checklist (credentials, shared network via `compose.eneo.yaml`, admission
sizing, upload caps).

### What Vemsa deliberately does not do

- **Multi-tenancy** — one consumer = one credential; fairness between eneo's tenants is
  eneo's problem, capacity between consumers is Vemsa's (`VEMSA_MAX_QUEUED_JOBS_PER_CLIENT`).
- **Interactive auth** — static server-to-server bearer tokens only; identity-aware
  ingress and OAuth live in front of Vemsa, not inside it.
- **Storage** — no archive of audio or transcripts beyond the retention window; the system
  of record is the consumer.
- **Trusting foreign timestamps** — the plausibility gate and `forced`-first policy exist
  because provider timelines have been observed compressed beyond human speaking rate; on
  the quality tiers a job that cannot be force-aligned fails loudly, and the rungs below
  `forced` are reachable only by deliberate configuration (`remote`,
  `VEMSA_DIARIZE_PREFER_ALIGN=false`), never by silent degradation.
