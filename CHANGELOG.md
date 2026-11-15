# Changelog

All notable changes to vemsa are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). A git tag `vX.Y.Z` publishes the
container images `ghcr.io/eneo-ai/vemsa:X.Y.Z` and `:X.Y` (GPU) and `:X.Y.Z-cpu` and
`:X.Y-cpu` (CPU), without the `v`.

## [0.2.0] - YYYY-MM-DD

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

- _To be added: word timing bounds (#9) and silent audio (#8)._

### Breaking

- No breaking changes identified. The minimum driver is unchanged according to NVIDIA, but
  minor version compatibility has exceptions: device code that is JIT-compiled from PTX, and
  CUDA features that require a newer driver, can fail on an older driver. Neither is shown
  to affect vemsa, and no run on a GPU has been made for this release.

**Rolling back:** the previous image is `ghcr.io/eneo-ai/vemsa:sha-<…>` (and
`sha-<…>-cpu`).

**Verified:** at build and source level only, not on a GPU: the lock file, an installation
of the GPU image's dependencies in a linux/amd64 container, and a probe that NVRTC 12.9.86
compiles for sm_103 while 12.8.93 rejects it.

[0.2.0]: https://github.com/eneo-ai/vemsa/releases/tag/v0.2.0
