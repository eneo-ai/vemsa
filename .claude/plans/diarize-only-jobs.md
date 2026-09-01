# Diarize-only jobs: speaker labels for an externally produced transcript

Companion plan on the Eneo side: `eneo/.claude/plans/vemsa-diarize-only.md`.

## Context

Eneo wants its own model registry (per-tenant providers, credentials, governance) to do
the ASR, and Vemsa to do only what its registry cannot: speaker diarization. Today the
only job kind runs the whole engine (`TranscriptionEngine.transcribe`), so a caller cannot
supply a transcript.

The pipeline is already separable: `Diarizer.diarize(audio_path) -> list[Turn]`
(`src/vemsa/pipeline/diarize.py:174`) needs only audio, and
`resolve_segments(words, segments, turns)` (`diarize.py:102`) plus `render_text`
(`src/vemsa/pipeline/render.py:9`) are pure functions that do not care where the words
came from. `HybridEngine` (`src/vemsa/pipeline/hybrid.py:39-58`) is a hand-composed
sequence of exactly these stages. This plan adds a second job kind that enters the
sequence after ASR.

Design goals: same job lifecycle (submit / poll / result), same `TranscriptionResult` out
so Eneo's client and the `[HH:MM:SS - HH:MM:SS] SPEAKER_00:` text contract do not move,
no engine (and therefore no `VEMSA_WHISPER_API_BASE`) required for diarize-only
deployments.

## 1. Request model (`src/vemsa/jobs/models.py`)

- Add `task: Literal["transcribe", "diarize"] = "transcribe"` to `JobRequest`.
- Add `words: list[Word] | None = None` and `segments: list[Segment] | None = None`
  (`Segment.words` may be empty on input; `speaker` is ignored on input).
- `model_validator(mode="after")`: `task == "diarize"` requires `words` or `segments`
  non-empty and `diarize` true; `task == "transcribe"` rejects transcript fields.
  `model` and `language` are accepted for `diarize` only as metadata to echo back in the
  result (`TranscriptionResult.model` / `.language`); default `model` to the string
  `"external"` when absent on a diarize job so the result never claims Vemsa's default
  model ran.
- Word timestamps are absolute seconds from the start of the uploaded audio; document
  this on the field. Reject `start > end` and negative values.
- `TranscriptionResult` unchanged. `Turn` stays internal (no design C leakage).

## 2. Submission (`src/vemsa/api/jobs.py`)

- `_job_from_multipart`: multipart values are strings, so before
  `JobRequest.model_validate(fields)` parse `words` / `segments` with `json.loads`
  when present, mapping `JSONDecodeError` to the existing 422 shape via
  `_validation_error`. Cap the transcript part size (new setting, see 5).
- JSON body path (`source_url` jobs): no change beyond the model; the transcript rides
  in the JSON.
- Keep `file` mandatory for `diarize` (pyannote needs audio). `source_url` works too.

## 3. Execution (`src/vemsa/jobs/queue.py`)

- Around the `asyncio.to_thread(self._engine.transcribe, ...)` call (`queue.py:131-137`)
  branch on `job.request.task`:
  - `transcribe`: unchanged.
  - `diarize`: `turns = self._diarizer.diarize(audio_path)` in a thread, then
    `segments = resolve_segments(words, plain_segments, turns)`, then
    `build_result(...)` / `render_text(segments)` into a `TranscriptionResult` with
    `duration_seconds` measured from the audio (reuse whatever the engines use;
    `soundfile`/`audioread` already present) and `model`/`language` echoed from the
    request.
- Give the queue a `Diarizer` dependency directly (constructor-injected, duck-typed like
  the engines already accept in `tests/test_whisper_api.py:45-56`). Factory: in
  `src/vemsa/pipeline/factory.py` build the `Diarizer` once and share it with the engine
  so a `hybrid`/`remote` deployment does not load pyannote twice.
- `Diarizer.load()` must be part of readiness for every engine tier now, including a
  new `VEMSA_ENGINE=diarize` tier (see 5) that constructs no ASR engine at all.
  `warm_up` for that tier is `Diarizer.load()`.
- Metrics: `JOBS_FINISHED` / `JOB_DURATION` already label by engine; add a `task` label
  or reuse the engine label with value `diarize` so dashboards separate the kinds.
- `fake` engine: diarize task returns the input words with `SPEAKER_00`/`SPEAKER_01`
  alternating per segment so Eneo can integration-test without torch.

## 4. Storage and hygiene

- `jobs.request_json` now carries transcripts. No migration (JSON column), but:
  - The retention purge (`VEMSA_RETENTION_HOURS`, `queue.py:281-309`) already deletes
    rows; confirm it deletes the request JSON, not only `result_json` and audio.
  - "Never log transcripts" (`docs/PRODUCTION.md`) now also covers request logging;
    audit any `logger.*` that prints `job.request`.
- Row size grows roughly 60 bytes per word; a 2 h recording is ~1 MB. Acceptable, but
  cap it (5).

## 5. Config (`src/vemsa/config.py`)

- `engine` literal gains `"diarize"`: `resolve_engine()` returns it verbatim;
  factory builds no ASR engine; `transcribe` jobs on such a deployment fail at
  submission with 422 `task 'transcribe' is not available on this deployment`.
- `VEMSA_MAX_TRANSCRIPT_BYTES` (default 8 MiB) for the words/segments part.
- Production validation (`config.py:124-130`): `diarize` tier requires `hf_token` and
  does not require `whisper_api_base`.

## 6. MCP (`src/vemsa/mcp/server.py`)

Out of scope. The MCP tools stay transcribe-only; note it in the docstring.

## 7. Tests

- `tests/test_assign_speakers.py`: unchanged (already covers the merge).
- New `tests/test_diarize_task.py`: multipart submit with a `words` JSON part; JSON
  submit with `source_url`; 422 on missing transcript, on `task=transcribe` with
  transcript fields, on oversized transcript, on bad JSON; queue branch with a
  `FakeDiarizer` produces speaker-labelled `text` and echoes `model`/`language`;
  `diarize` engine tier readiness without `VEMSA_WHISPER_API_BASE`.
- `tests/test_hybrid.py` / `test_whisper_api.py`: assert the shared `Diarizer` is
  constructed once.

## 8. Docs

- `README.md` API section: the `task` field, transcript part format, example curl.
- `docs/PRODUCTION.md`: the `diarize` tier, sizing (CPU roughly real time; GPU for
  fan-in), retention now covering transcripts.
- `.claude/plans/eneo-integration-handover.md`: add the new contract facts to the
  change-control list (task field, absolute timestamps, `model="external"` echo).

## Verification

1. `uv run pytest` (unit, no torch) green including the new file; `ruff check`.
2. Local: `VEMSA_ENGINE=fake`, submit `curl -F file=@a.mp3 -F task=diarize
   -F 'words=[{"word":"hej","start":0.0,"end":0.4}]'`, poll, fetch result; `text` has
   speaker prefixes and `model == "external"`.
3. GPU box: `VEMSA_ENGINE=diarize` with a real HF token, a 10 min two-speaker file and a
   transcript produced by Eneo's adapter; compare speaker turns against the `hybrid`
   tier's output on the same file.
4. `readyz` on the `diarize` tier without `VEMSA_WHISPER_API_BASE` set.
