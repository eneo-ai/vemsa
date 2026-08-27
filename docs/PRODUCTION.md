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

## Container images

CI publishes images to `ghcr.io/eneo-ai/tolka` on every push to `main` and on `v*` release
tags (`.github/workflows/docker.yaml`). One image serves both the `api` and `worker`
services; two flavors are built:

| Flavor | Tags | Torch | Platforms | For |
| --- | --- | --- | --- | --- |
| GPU (default) | `latest`, `main`, `vX.Y.Z`, `X.Y`, `sha-<commit>` | CUDA 12.8 wheels | amd64 | hosts with an NVIDIA GPU (the `local`/`hybrid` engines, fast diarization) |
| CPU | same tags with `-cpu` suffix (`latest-cpu`, …) | CPU wheels | amd64, arm64 | GPU-less hosts, `remote` engine deployments, Apple-silicon servers |

Both flavors bake in every ML extra (`diarize align local`), so `TOLKA_ENGINE` alone decides
behavior at runtime. Slimmer builds (e.g. `ML_EXTRAS="diarize"` for remote-only) are a local
`docker build --build-arg` away but are not published.

### GPU or CPU?

The GPU image only *uses* a GPU when the host provides one: it needs the NVIDIA driver and
`nvidia-container-toolkit` on the host, plus the device reservation already present on the
`worker` service in `compose.yaml`. Torch wheels bundle CUDA/cuDNN, so no CUDA base image or
host CUDA install is required.

For a CPU-only deployment, use the `-cpu` image and drop the GPU reservation with an
override file:

```yaml
# compose.cpu.yaml — !reset (Compose v2.24+) removes the merged-in GPU reservation;
# a plain empty mapping would merge and keep it
services:
  worker:
    deploy: !reset {}
```

```bash
TOLKA_IMAGE=ghcr.io/eneo-ai/tolka:latest-cpu \
  docker compose -f compose.yaml -f compose.cpu.yaml up --no-build
```

Expect CPU diarization at roughly real time and plan job deadlines accordingly; the
`remote` engine tier keeps transcription itself fast by offloading whisper.

### Selecting an image

`compose.yaml` names `${TOLKA_IMAGE:-ghcr.io/eneo-ai/tolka:latest}` on both services while
keeping `build: .`, so the default `docker compose up --build` workflow still builds from
source. To run a published image instead:

```bash
export TOLKA_IMAGE=ghcr.io/eneo-ai/tolka:v0.1.0        # or @sha256:… (preferred, see below)
docker compose pull api worker
docker compose up --no-build
```

Per the release gate, pin the deployed image by digest in the target environment rather
than by mutable tag.

## Authentication and exposure

Production credentials must be named:

```text
TOLKA_API_TOKENS=eneo=long-random-value,automation=another-long-random-value
```

The name becomes `client_id` and owns the job. Tokens are internal server-to-server
credentials, not an interactive OAuth implementation. Keep the service on a private network
or behind an identity-aware API gateway. TLS terminates at that ingress. Never log tokens,
transcripts, complete source URLs, or webhook payloads.

## Diarize-only jobs and the optional diarize tier

Every engine tier serves both job kinds: `task=transcribe` runs the full engine and
`task=diarize` labels speakers on a caller-supplied transcript, chosen per request. The
expected topology is one full deployment where the consumer's request says what it wants.

`TOLKA_ENGINE=diarize` is an optional lockdown for a deployment that should never
transcribe: no ASR engine is constructed, `TOLKA_WHISPER_API_BASE` is not
required, and `task=transcribe` submissions are rejected at admission with 422. The tier
requires `HF_TOKEN` in production (gated pyannote models). Sizing: pyannote runs at roughly
real time on CPU — fine for short recordings; use a GPU host when fan-in or recording length
makes that untenable. Caller transcripts ride in `jobs.request_json`, so the retention purge
and the "never log transcripts" rule now cover request payloads as well as results;
`TOLKA_MAX_TRANSCRIPT_BYTES` (default 8 MiB) caps the accepted transcript size.

## Consumer integrations (eneo)

Eneo integrates Tolka as the engine for its flow `transcribe_only` steps. Tolka exposes
authenticated readiness, coarse job stages, queue position, and idempotent cancellation.
Eneo uploads the original audio as multipart to `POST /v1/jobs`, polls
`GET /v1/jobs/{id}`, cancels through `DELETE /v1/jobs/{id}`, and passes `result.text`
verbatim into the flow output. Deployment checklist:

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
6. **Readiness**: call authenticated `GET /v1/health/ready` before enabling the integration.
   Treat `queue_accepting_jobs=false` as capacity degradation and `503` as unavailable.
7. **Latency**: characterize worst-case job latency for the deployment's engine tier, GPU,
   and maximum file size against eneo's poll deadline. A job which crashes the worker is
   re-leased without an attempt cap, so the consumer deadline remains the backstop.

The response contract eneo depends on (multipart part name `file`, status/stage enums,
cancellation semantics, `TranscriptionResult`, and the rendered
`[HH:MM:SS - HH:MM:SS] SPEAKER_00:` line format) is change-controlled. Shape changes must
move in lockstep with eneo's `RemoteTranscriptionClient` and its contract tests.

## Cancellation and recovery

`DELETE /v1/jobs/{id}` is client-scoped and idempotent. Active jobs are atomically moved to
`cancelled`, their lease is released, and an optional cancellation webhook is inserted in
the same transaction. Queued upload files are deleted by the API. A worker already inside a
blocking ML stage may finish that stage, but its next stage update detects cancellation and
the job store refuses late results or failures. The worker then deletes its temporary audio.

For stuck jobs, first inspect `status`, `stage`, `updated_at`, worker heartbeat, queue depth,
and lease expiry. Cancel work that is no longer wanted; restart an unhealthy worker and let
the lease expire for work that should retry. Do not edit job rows manually during ordinary
recovery.

Roll out the Tolka changes before updating consumers. Keep the consumer integration disabled
until authenticated readiness succeeds. Roll back by disabling consumer submissions,
cancelling unwanted active jobs, and restoring the previous pinned image; existing terminal
results remain readable until retention removes them.

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
- `/v1/health/ready` is authenticated and reports service version, database/worker readiness,
  queue admission state, and queue depth.
- `/metrics` exposes authenticated Prometheus data.

Alert at minimum on oldest queued-job age, queue depth, queue rejection rate, failure and
cancellation ratio, per-stage duration, webhook exhaustion, missing worker heartbeats, disk
usage, PostgreSQL availability, GPU OOM, and processing real-time factor. Logs are JSON by
default and include `request_id`, `job_id`, `client_id`, engine, task, stage, attempt, status,
and duration where applicable. They must never contain credentials, audio, transcripts,
participant names, complete source URLs, or webhook payloads.

## Backups and retention

Back up PostgreSQL with the organization's standard encrypted backup and point-in-time recovery
process. Exercise restore tests. The `/data` volume contains temporary source material and is
not a substitute for a database backup. Results are deleted after `TOLKA_RETENTION_HOURS`;
confirm that setting against the organization's records and privacy policy.

## Release gate

Before deploying a service or image revision:

1. Build from the committed `uv.lock` and scan the image and SBOM.
2. Pin the deployed image by digest in the target environment.
3. Run the complete unit, REST, MCP, store, queue, contract, and security test suites.
4. Run REST and MCP smoke tests through the real ingress, using a generated WAV and the fake
   engine only outside production.
5. Verify authenticated readiness, queue saturation, stage progression,
   cancellation races, graceful worker termination, lease recovery, and webhook retry.
6. Verify backup, restore, retention cleanup, and rollback against the pinned image.

This release gate verifies service correctness and failure handling, not transcription or
diarization accuracy. WER, speaker accuracy, and real-world correction burden remain
unverified until representative, consented audio becomes available. The local and hybrid ML
integrations still contain `GPU-VERIFY` markers; CI does not certify their model quality.
