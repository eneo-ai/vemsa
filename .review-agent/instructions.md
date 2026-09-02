# Vemsa review instructions

## Engineering principles

- Preserve one clear owner for job state, engine selection, and quality gates.
- Prefer explicit failure over silent quality degradation in transcription,
  alignment, and diarization paths.
- Keep asynchronous lifecycle transitions idempotent and safe under retries,
  cancellation, and worker lease loss.
- Avoid loading large audio, model, or transcript data into memory when a
  bounded or streaming path is available.

## Review focus

- Check that changes preserve authentication, per-client job ownership, and
  source-audio cleanup.
- Trace changes to timestamp alignment and speaker attribution through their
  tests and documented quality contract.
- Treat production GPU requirements and optional local development behavior as
  separate concerns.

## Communication style

- Be direct, constructive, and concise.
- Explain the verified failure path and the smallest complete correction.
