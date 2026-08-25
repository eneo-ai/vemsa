# Eneo integration handover

Eneo (adjacent repo) is integrating Tolka as the transcription engine for its flow
`transcribe_only` steps: eneo submits the original audio bytes as a multipart job to
`POST /v1/jobs` (`diarize=true`, `language` from flow config: `sv|en|auto`), polls
`GET /v1/jobs/{id}` every ~5s from a dedicated worker, and fetches
`GET /v1/jobs/{id}/result`, passing `result.text` verbatim into the flow output and
`duration_seconds` into usage accounting. Eneo-side plan:
`eneo/.claude/plans/tolka-flow-transcription.md`.

## What eneo's v1 needs from Tolka (config/deployment only — no code changes identified)

1. **Token provisioning**: a named credential for eneo, e.g. `TOLKA_API_TOKENS=eneo=<secret>`; eneo stores it as `FLOW_TRANSCRIPTION_SERVICE_API_KEY`.
2. **Deployment reachability**: the eneo flow execution worker must reach Tolka's API. Tolka's compose binds `api` to `127.0.0.1` — expose it on the shared network for the eneo deployment (and the devcontainer for local dev). GPU worker sizing per Tolka's compose (`TORCH_VARIANT`, `ML_EXTRAS`).
3. **Admission-control sizing**: `TOLKA_MAX_QUEUED_JOBS_PER_CLIENT` (default 10) bounds eneo's concurrent submissions; all eneo tenants share one `client_id`, and eneo's flow concurrency is 4 runs/tenant — size for multi-tenant fan-in. 429 is returned before the body is read; eneo treats it as retryable.
4. **Upload limit**: `TOLKA_MAX_AUDIO_BYTES` (default 2 GiB) must be ≥ eneo's max audio upload size (binds on the original compressed file — eneo sends originals, not wav).
5. **Retention**: `TOLKA_RETENTION_HOURS` (default 72) is ample; eneo fetches results within the run (≤1h).
6. **Engine tier**: per deployment (`local` on a GPU box, or `hybrid` with `TOLKA_WHISPER_API_BASE` pointing at existing hosted whisper). Tolka's own maturity markers: `local`/`hybrid` carry GPU-VERIFY(milestone-2) flags; `remote` is fully CI-exercised.
7. **Expected max job latency** for a 2 GiB file should be characterized so eneo's 3300s poll deadline can be validated (Tolka worker-crash retries extend wall time unboundedly — see gap 6 below).

## Known gaps in Tolka that eneo v1 works around (candidate future work, priority order)

1. **No progress signal** — no queue position, percentage, or ETA on `GET /v1/jobs/{id}`. Eneo shows an indeterminate running state. Future: add `queue_position`/coarse progress.
2. **No cancel endpoint** — an aborted/timed-out eneo step leaves the Tolka job running to completion (wasted GPU). Future: `DELETE /v1/jobs/{id}`.
3. **No speaker-count hints** — `min_speakers`/`max_speakers` passthrough to pyannote is already marked future in `diarize.py`.
4. **Webhook path unused by v1** — Tolka's transactional outbox + HMAC webhooks pair with a future eneo async-checkpoint wait model. No Tolka change needed; noting the pairing.
5. **Per-end-user attribution absent by design** — `client_id` == token name; eneo attributes usage per tenant/run on its side.
6. **`attempt` unbounded** — a job that crashes the worker is re-leased forever. Consider a max-attempts cap so poisoned audio fails instead of looping (this also protects eneo's poll deadline).

## Contract facts eneo's implementation depends on (change control)

- `POST /v1/jobs` multipart part is literally named `file`; other form fields validate through `JobRequest` (`language` closed to `sv|en|auto`, `diarize` with string coercion, `model`). 429 pre-body, 413 oversize, 401 auth.
- Status enum `queued|running|completed|failed`; result via `GET /v1/jobs/{id}/result` (409 pre-completion, 404 cross-client/unknown).
- `TranscriptionResult{language, duration_seconds, model, text, segments[{start,end,speaker,text,words[]}]}`; `text` is the rendered speaker-labeled transcript — **eneo passes it verbatim into flow output**, so the `[HH:MM:SS - HH:MM:SS] SPEAKER_00:` line format is user-visible contract.
- Job errors are sanitized (`ExcType: processing failed`) — eneo expects no diagnostic detail.
- Diarize-only jobs (added after v1): `task=diarize` on `POST /v1/jobs` with a caller-supplied
  transcript (`words` and/or `segments` as JSON; multipart parts or JSON body fields).
  Timestamps are absolute seconds from the start of the uploaded audio; 0 <= start <= end
  is validated. Transcript size is capped by `TOLKA_MAX_TRANSCRIPT_BYTES` (default 8 MiB,
  413 over it). The result is the same `TranscriptionResult` shape; `model` is echoed from
  the request and defaults to `"external"`. `TOLKA_ENGINE=diarize` deployments reject
  `task=transcribe` with 422. `diarize=false` with `task=diarize` is a 422.
  Version-skew hazard: a pre-task Tolka deployment silently ignores the unknown fields and
  runs the job as a plain transcription — clients should assert `result.model == "external"`
  (or their echoed model) before trusting a diarize-only result.
- If any of these change shape, eneo's `RemoteTranscriptionClient` (and its contract tests) must move in lockstep.
