# Production operations

## Runtime shape

Production Compose runs three services:

1. `api` serves REST and FastMCP without loading ML models.
2. `worker` claims leased jobs from PostgreSQL and owns the GPU pipeline.
3. `postgres` stores jobs, results, worker heartbeats, and webhook outbox events.

SQLite remains the default for tests and single-process development. Do not place its WAL
database on NFS and do not scale SQLite-backed API processes. PostgreSQL claims use row locks
with `SKIP LOCKED`, expiring leases, and heartbeats so multiple workers do not intentionally
claim the same job.

Audio files live on the shared `/data` Docker volume and are deleted after a terminal job.
That limits the supplied Compose deployment to one Docker host. Use private object storage
before scheduling workers on multiple machines.

## Authentication and exposure

Production credentials must be named:

```text
TOLKA_API_TOKENS=eneo=long-random-value,automation=another-long-random-value
```

The name becomes `client_id` and owns the job. Tokens are internal server-to-server
credentials, not an interactive OAuth implementation. Keep the service on a private network
or behind an identity-aware API gateway. TLS terminates at that ingress. Never log tokens,
transcripts, complete source URLs, or webhook payloads.

## Consumer integrations (eneo)

Eneo integrates Tolka as the engine for its flow `transcribe_only` steps: it uploads the
original audio as multipart to `POST /v1/jobs` (`diarize=true`, `language` from flow config),
polls `GET /v1/jobs/{id}` every ~5s from a dedicated worker with a 3300s deadline, and passes
`result.text` verbatim into the flow output. Deployment checklist:

1. **Credential**: provision a named token, `TOLKA_API_TOKENS=eneo=<long-random-value>`;
   eneo stores it as `FLOW_TRANSCRIPTION_SERVICE_API_KEY`.
2. **Reachability**: the Compose file publishes the API on loopback only. Set `TOLKA_BIND`
   to a private interface reachable by eneo's flow execution worker, or use
   `compose.eneo.yaml` to attach the API container to a shared Docker network. For local
   development against eneo's devcontainer, `TOLKA_BIND=0.0.0.0` makes the port reachable
   via `host.docker.internal` (loopback-published ports are not).
3. **Admission control**: all eneo tenants share the one `eneo` client id, so
   `TOLKA_MAX_QUEUED_JOBS_PER_CLIENT` (default 10) bounds eneo's total concurrent
   submissions across tenants. Eneo runs up to 4 concurrent flow runs per tenant; size the
   per-client limit (and `TOLKA_MAX_QUEUED_JOBS`) for the expected multi-tenant fan-in.
   Eneo treats the pre-body 429 as retryable.
4. **Upload cap**: `TOLKA_MAX_AUDIO_BYTES` (default 2 GiB) binds on the original compressed
   file eneo sends; keep it at or above eneo's maximum audio upload size.
5. **Retention**: the 72-hour default is ample; eneo fetches results within the run (≤1h).
6. **Latency**: characterize worst-case job latency for the deployment's engine tier, GPU,
   and maximum file size against eneo's 3300s poll deadline. Note that a job which crashes
   the worker is re-leased without an attempt cap, so pathological inputs can exceed any
   deadline; eneo's poll deadline is its backstop.

The response contract eneo depends on (multipart part name `file`, the status enum, the
`TranscriptionResult` shape, and the rendered `[HH:MM:SS - HH:MM:SS] SPEAKER_00:` line
format of `text`) is change-controlled: shape changes must move in lockstep with eneo's
`RemoteTranscriptionClient` and its contract tests.

## Outbound network policy

Tolka rejects URL user information, non-HTTP schemes, private addresses by default, and an
unapproved hostname when an allowlist is configured. Each source redirect is checked.
Webhooks require HTTPS by default.

Application checks cannot fully prevent DNS rebinding between validation and connection.
Production must also restrict container egress: allow DNS, the configured Whisper endpoint,
approved source hosts or an outbound proxy, approved webhook destinations, and required model
registries. Block cloud metadata, link-local, cluster-control, and internal administration
networks.

## Webhook delivery

Completion and failure events are inserted transactionally with the terminal job update.
Workers claim outbox rows, retry with exponential backoff, and retain exhausted events for
operator inspection until the parent job's retention cleanup.

When `TOLKA_WEBHOOK_SIGNING_SECRET` is set, requests include:

- `X-Tolka-Timestamp`
- `X-Tolka-Signature-256: sha256=<hex digest>`

The digest is HMAC-SHA256 over `<timestamp>.<raw request body>`. Consumers should reject stale
timestamps and compare signatures in constant time.

## Health and telemetry

- `/livez` checks that the API process can answer.
- `/readyz` checks PostgreSQL and a recent worker heartbeat.
- `/metrics` exposes authenticated Prometheus data.

Alert at minimum on oldest queued-job age, queue depth, job failure ratio, webhook exhaustion,
missing worker heartbeats, disk usage, PostgreSQL availability, GPU OOM, and processing
real-time factor. Logs are JSON by default and include `request_id`, `job_id`, `client_id`,
engine, attempt, status, and duration where applicable.

## Backups and retention

Back up PostgreSQL with the organization's standard encrypted backup and point-in-time recovery
process. Exercise restore tests. The `/data` volume contains temporary source material and is
not a substitute for a database backup. Results are deleted after `TOLKA_RETENTION_HOURS`;
confirm that setting against the organization's records and privacy policy.

## Release gate

Before deploying a model or image revision:

1. Build from the committed `uv.lock` and scan the image and SBOM.
2. Pin the deployed image by digest in the target environment.
3. Run REST and MCP smoke tests through the real ingress.
4. Transcribe representative Swedish recordings on the target GPU.
5. Verify language auto-detection, alignment, diarization, fallback behavior, and OOM recovery.
6. Test graceful worker termination, lease recovery, webhook retry, backup, and restore.

The local and hybrid ML integrations still contain `GPU-VERIFY` markers. Those checks require
real model licenses, provider access, representative recordings, and production-class GPU
hardware; CI cannot certify them.
