# Eneo handover: correction round-trip (`task=align`) and label-preserving re-diarize

Companion to [eneo-integration-handover.md](eneo-integration-handover.md). Vemsa side is
implemented on `main` (uncommitted at the time of writing; lands in the next
`ghcr.io/eneo-ai/vemsa` publish). Eneo's transcript work lives on the flows line
(`b563fc2ee`, `refactor/flows-tidy-ai-builder`), which still calls the service "tolka".

## Why

A human corrects a misheard word or moves a sentence to another speaker in eneo's
transcript player. Today nothing is realigned: eneo keeps corrections as char-range
overlays (`flow_transcript_corrections`) on an immutable segment array, timestamps never
change, and no word timings are stored. Two user-visible consequences:

- A partial-sentence speaker reassignment renders as two lines sharing one
  `[start - end]` prefix, because no timestamp exists for the split point.
- Seek is line-granular; the words eneo could highlight or seek to are discarded.

Resubmitting the corrected text as `task=diarize` was the only option, and it re-ran
pyannote, renumbered speakers arbitrarily, and ignored the caller's labels — a corrected
transcript came back with scrambled names.

## What Vemsa now offers

### 1. `task=align` — re-time a corrected, speaker-labelled transcript

Audio + corrected segments in; the same `TranscriptionResult` out with word timestamps
re-derived from the audio. No ASR, no diarization, one alignment pass (seconds of GPU
time, not a pipeline run). Accepted on every engine tier, including `VEMSA_ENGINE=diarize`.

```
POST /v1/jobs   (multipart, same auth/admission/lifecycle/webhooks as every job)
  file=<original audio>          # or JSON body with source_url — eneo's signed URL works
  task=align
  language=sv
  segments=[
    {"start": 12.3, "end": 15.1, "speaker": "SPEAKER_00", "text": "Hej och välkomna."},
    {"start": 15.1, "end": 19.8, "speaker": "SPEAKER_01", "text": "Vi ses imorgon"},
    {"start": 15.1, "end": 19.8, "speaker": "SPEAKER_00", "text": "ja det gör vi."}
  ]
```

Guarantees eneo can rely on:

- `speaker` and `text` come back **verbatim**, segments in time order. Labels are opaque
  strings — send `SPEAKER_NN` or real names, whatever eneo stores.
- Every segment's `start`/`end` is tightened to its first/last aligned word, and
  `segments[].words[]` is populated with `{word, start, end, probability}` (CTC score).
- `alignment` is always `"forced"`; `model` is echoed (default `"external"`).
- **Segment windows are the alignment anchors**: send the windows from the previous
  result. A sentence split between two speakers is sent as two segments over the *same*
  window (as above); Vemsa aligns the whole sentence once and the audio decides where the
  split lands. Consecutive segments overlapping or within 0.5 s are aligned together;
  every window gets 0.5 s of slack, so tight windows are fine.
- A segment with no alignable text (e.g. `"…"`) keeps its input window and gets no words.
- Validation (422): `segments` required and non-empty with text; `words` rejected;
  `num_speakers`/`min_speakers`/`max_speakers` rejected; `vocabulary` rejected;
  `diarize=false` rejected; `0 <= start <= end` per segment. Transcript size cap
  `VEMSA_MAX_TRANSCRIPT_BYTES` (413) applies.
- Stages observed while polling: `aligning → finalizing`.

**The "does this edit still fit the audio" signal.** On the `forced` rung, a word with
`probability` exactly `0.0` was *interpolated, not aligned*: the aligner could not fit a
window's text to its audio (edited text much longer than the window, or characters the
CTC vocabulary lacks — digits and symbols are not spelled out) and spread the words evenly
over the window. Eneo should flag such words for review rather than trust their times.
Vemsa logs each such window (`align.interpolated`), counts them in
`vemsa_alignment_interpolated_words_total`, and `VEMSA_ALIGN_MAX_INTERPOLATED_SHARE`
(default `1.0` = never) fails the job above a share — the failure message names the knob
and passes through unsanitized, like the `VEMSA_MIN_ALIGNMENT` floor.

### 2. `task=diarize` now honours caller speaker labels

Caller segments may carry `speaker`. They never decide attribution (the audio does), but
each new pyannote cluster is renamed after the caller label holding most of its overlap
with labelled speech (`VEMSA_ATTR_RELABEL_MIN_SHARE`, default 0.5). Names a human
already assigned survive a re-run; two clusters may land on one label (the diarizer
over-split a speaker the human had merged); a cluster matching no label gets a fresh
`SPEAKER_NN` that collides with nothing. Unlabelled segments behave exactly as before.

This is the primitive for a "Rerun speaker identification" action on an already-corrected
transcript: send the current labelled segments as
`task=diarize`, optionally with `num_speakers`, and the names come back intact.

## Version-skew

- A pre-align Vemsa rejects `task=align` with **422** (`task` is a closed enum), never a
  silent plain transcription — no `model` assertion trick needed, but eneo should treat
  that 422 as "feature unavailable", not as a payload bug.
- A pre-relabel Vemsa silently ignores `speaker` on `task=diarize` input and returns
  fresh `SPEAKER_NN` labels. If eneo depends on preservation, check that the returned label
  set intersects the one it sent.

## Recommended eneo flow

1. **Persist words.** `RemoteTranscriptionResult.segments[].words` is already on the wire
   and dropped. The 256 KiB `MAX_SEGMENTS_BYTES` cap on `input_payload_json` will not hold
   word arrays for hour-long audio (~10 words/s × ~60 bytes); store them file-backed or in
   their own table, keyed by `(flow_run_id, step_id, attempt)`.
2. **Realign on approval, not on keystroke.** The natural hook is
   `flow_transcript_corrections_propagation.py`, which already folds corrections into the
   checkpoint text on review approval. Build the `task=align` payload from the raw
   segments with corrections and `speaker_edits` applied (a partial-sentence
   reassignment becomes two segments over the original window), reuse the signed
   input-file URL as `source_url` or re-upload, and store the returned segments + words as
   the **new raw segment array** for downstream steps and the finished-run player.
3. **Expect the overlay set to go stale by design.** The corrections table is keyed to
   `segments_hash`; a realign produces a new hash. Either archive the overlay set as
   history (the `original` text is already retained for audit) or reset it. Do not try to
   re-anchor overlays onto the realigned segments.
4. **Flag interpolated words.** Render `probability == 0.0` (and optionally low scores)
   as uncertain in the player; that is the reviewer's work queue.
5. **"Rerun speaker identification"** from the player: current labelled segments as
   `task=diarize` (+ `num_speakers` if the user sets it); merge the result the same way as
   a realign. Speaker *names* stay a render-time overlay (`speaker_mapping` step); labels
   round-trip as whatever eneo sends.
6. **Contract tests.** Add `task=align` fixtures to `RemoteTranscriptionClient`'s tests:
   submit with segments, 422 on `words`, result carries `alignment: "forced"`, speakers
   verbatim, and a `probability: 0.0` word marked uncertain.

Eneo-side anchors (flows branch): client
`backend/src/eneo/flows/runtime/remote_transcription.py` (`submit()` already sends
`task`, `segments`; add `speaker` to the segment serializer); storage
`backend/src/eneo/flows/runtime/transcription.py:163` (segment shape, no `words`);
corrections domain `backend/src/eneo/flows/domain/transcript_corrections.py` and
`flows/application/flow_transcript_corrections_propagation.py`; player
`frontend/apps/web/src/lib/features/flows/components/TranscriptPlayer.svelte`
(`findActiveSegmentIndex` is where word-level highlight would go).

## Contract facts (change control, additive)

- New `task` value `align`; `Segment.speaker` honoured on input for `diarize` and `align`.
- Result shape unchanged. `Word.probability == 0.0` on the `forced` rung now has a defined
  meaning (interpolated); non-zero scores are unchanged.
- New settings: `VEMSA_ATTR_ALIGN_MERGE_GAP_S` (0.5), `VEMSA_ATTR_ALIGN_WINDOW_PAD_S` (0.5),
  `VEMSA_ATTR_RELABEL_MIN_SHARE` (0.5), `VEMSA_ALIGN_MAX_INTERPOLATED_SHARE` (1.0).
- Rendered `[HH:MM:SS - HH:MM:SS] SPEAKER:` text format unchanged; a realigned split now
  renders as two lines with two distinct time prefixes.

## Verification status on the Vemsa side

- 214 unit/integration tests green (no ML stack; PostgreSQL-backed suite included).
- Real CPU end-to-end with the Swedish wav2vec2 aligner on a synthesized clip: a
  corrected word landed where the misheard one was, a split sentence got two distinct
  windows at the pause, zero interpolated words, labels preserved through a re-diarize
  with swapped cluster numbering.
- Found and worked around an easyaligner 0.3.3 defect: a word ending exactly at its
  window's end gets a garbage end time (stretched to the 30 s chunk end). Align windows
  are padded; eneo's `task=diarize` chunk windows already carry slack and are unaffected.
- GPU runs of both tasks join the existing `GPU-VERIFY` gate in `docs/PRODUCTION.md`.
