# Vemsa quality context

Vemsa exposes REST and MCP adapters over the same durable job engine. FastMCP
is a protocol adapter, not a second queue or transcription implementation.

Production quality tiers use local CTC forced alignment as the source of word
timestamps and local pyannote diarization as the source of speakers. When a
path requires forced alignment, a missing or failed aligner must fail the job
rather than silently return a coarser timestamp rung. The explicit `remote`
engine is the exception: it may trust provider timestamps when the deployment
accepts and enforces that quality contract.

Jobs are owned by the authenticated client identity. Submit, poll, cancel,
result, retention, and webhook changes must preserve that boundary as well as
idempotent cancellation and worker-lease behavior.

Relevant human documentation lives in `README.md`, `docs/ARCHITECTURE.md`, and
`docs/PRODUCTION.md`. Tests under `tests/` are the executable contract.
