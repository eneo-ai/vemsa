# Changelog

All notable changes to vemsa are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). A git tag `vX.Y.Z` publishes the
container images `ghcr.io/eneo-ai/vemsa:X.Y.Z` and `:X.Y` (GPU) and `:X.Y.Z-cpu` and
`:X.Y-cpu` (CPU), without the `v`.

## [Unreleased]

### Fixed

- A failed webhook delivery is planned again after its backoff, as documented. Before, the
  retry was never scheduled, and the worker delivered no more webhooks until it was
  restarted. A delivery attempt that fails in any other way also counts as a failed attempt,
  so a webhook that keeps failing ends after `VEMSA_WEBHOOK_MAX_ATTEMPTS` recorded failed
  attempts; a destination that breaks a rule ends at the attempt that finds it. An error
  while reading or updating the webhook outbox no longer stops delivery: the worker logs the
  error type, and the job id when it is known, waits one poll interval and goes on. A
  webhook can be delivered more than once, for example when its delivery cannot be
  confirmed; receivers recognise a duplicate by its `job_id` and `status`.
- A webhook whose destination is rejected ends with the stored error `destination rejected`,
  and the worker's log line says only that the destination for the job was rejected.
- A webhook host name that the destination check cannot look up, which can be a temporary
  DNS failure, is a failed attempt: it is retried with backoff, the destination is checked
  again on each attempt, and a host that never resolves uses up its attempts and ends as
  `exhausted`. The stored error is `UnresolvableHostError`; in
  `vemsa_webhook_deliveries_total` each planned retry counts as `retry` and the last attempt
  as `exhausted`. With `VEMSA_ALLOW_PRIVATE_URLS=true` the check does not look up the host,
  and a lookup that fails while connecting is a failed attempt like any other. A destination
  that breaks a rule, such as the scheme, the allowlist or a non-public address, ends the
  webhook at the attempt that finds it. Fetching source audio is unchanged: when its
  destination check cannot look up the host, the job fails with
  `ForbiddenUrlError: processing failed`.

## [0.2.0] - 2026-11-15

### Added

- The GPU image supports NVIDIA Blackwell B300 (sm_103). On CUDA 12.8, torch's runtime
  kernel compiler (NVRTC 12.8) rejected that architecture
  ([#6](https://github.com/eneo-ai/vemsa/issues/6)); NVRTC 12.9 compiles for it.

### Changed

- The GPU image uses CUDA 12.9 wheels (`torch 2.8.0+cu129`, NVRTC 12.9.86) instead of CUDA
  12.8. The bundled NVIDIA CUDA libraries move to their 12.9 releases; cuDNN, cuSPARSELt and
  NCCL are unchanged. The CPU image is unchanged.
  - The minimum host driver is unchanged: `>=525.60.13`, NVIDIA's minimum for CUDA 12.x
    minor version compatibility, the same for 12.8 and 12.9.
  - A B300 needs a driver release that recognises the GPU. The only driver reported to work
    with a B300 is the 595 series ([#6](https://github.com/eneo-ai/vemsa/issues/6)).
- `service_version` in the readiness response reports `0.2.0`.

### Fixed

- A word cut off by an alignment window boundary no longer gets a span of up to 30 seconds
  across later speech and past the end of the audio. Where vemsa knows the alignment
  windows, each word stays inside the window it was aligned in; on the local engine's
  `transcribe` task only the next-word bound below applies. A word not placed by the audio
  (`probability: 0.0`) ends no later than the next word starts. The response shape is
  unchanged, but zero-duration words are more common and the number of lines can change. An
  alignment segment whose speech id matches no window now fails the job.
  ([#9](https://github.com/eneo-ai/vemsa/pull/9))
- On `hybrid` and `remote`, a provider answer with blank `text` and no words or segments,
  such as for a short silent last chunk of a long recording, now completes as an empty
  transcript (`text` `""`, no segments, `alignment` `null`) instead of failing the job, also
  with `VEMSA_MIN_ALIGNMENT` set; alignment and diarization are skipped for it. Operations:
  `vemsa_job_alignment_total{alignment="none"}` now also counts these silent chunks, which
  end `completed`, so a rise in `none` on `hybrid` can mean more silent chunks, not worse
  quality. ([#8](https://github.com/eneo-ai/vemsa/pull/8))

### Breaking

- No breaking changes identified. The minimum driver is unchanged according to NVIDIA, but
  minor version compatibility has exceptions: device code that is JIT-compiled from PTX, and
  CUDA features that require a newer driver, can fail on an older driver. Neither is shown
  to affect vemsa, and no run on a GPU has been made for this release.

**Rolling back:** the previous image is `ghcr.io/eneo-ai/vemsa:sha-c92e94f` (and
`sha-c92e94f-cpu`).

**Verified:** the CUDA change at build and source level only, not on a GPU: the lock file,
an installation of the GPU image's dependencies in a linux/amd64 container, and a probe that
NVRTC 12.9.86 compiles for sm_103 while 12.8.93 rejects it. The fixes are covered by unit
tests; the acceptance tests that need the real aligner and models are skipped in CI.

[0.2.0]: https://github.com/eneo-ai/vemsa/releases/tag/v0.2.0
