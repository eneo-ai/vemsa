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
