# Speaker review, version 1

Status: implemented in Vemsa; Eneo and Lyssna consumer work is described in the
linked handovers. This is an opt-in signal that attribution needs review. It is
not a speaker-count fix, a calibrated confidence score, or human confirmation.

## Request and compatibility

Add `include_speaker_review: true` to JSON `POST /v1/jobs`, or send the multipart
field `include_speaker_review=true`. Supported for `task=transcribe` with
`diarize=true`, and for `task=diarize`. The default is false. Exact
`num_speakers`, or a `min_speakers`/`max_speakers` range, continue to work as before.
An incomplete attendee/name list must not become an exact speaker count.

Without the flag, assignments, segment grouping, and rendered text retain their
previous behavior. Additive fields serialize as `speaker_review: null`,
`speaker_attribution: null`, and `overlap_ids: []`; old stored jobs/results also
parse with these defaults. Consumers must tolerate additive JSON fields.

`task=align` rejects the flag and non-empty review metadata on input, including
segment attribution states. An align-only operation cannot recompute diarization
evidence on the new timeline. Do not strip these fields to bypass the rejection.
Consumers need a deliberate re-diarization/review workflow before enabling
realignment for reviewed transcripts. A new `task=diarize` recomputes model
metadata; it does not preserve human review decisions.

## Result

```json
{
  "speaker_review": {
    "version": 1,
    "overlap_detection": "available",
    "overlaps": [
      {"id": "overlap_0000", "start": 1.0, "end": 1.5, "detected_speaker_count": 2}
    ]
  },
  "segments": [
    {
      "start": 1.0,
      "end": 1.5,
      "speaker": "SPEAKER_00",
      "text": "ses",
      "words": [{"word": "ses", "start": 1.0, "end": 1.5, "probability": null}],
      "speaker_attribution": "provisional",
      "overlap_ids": ["overlap_0000"]
    }
  ]
}
```

The example is a fragment of the existing `TranscriptionResult`, not a new
endpoint. Word probability keeps its existing ASR/alignment semantics.

| Field/state | Meaning |
| --- | --- |
| `speaker_review: null` | Review metadata was not requested or the result predates this contract. |
| `available`, empty `overlaps` | The regular diarization track was inspected and no overlap was detected. This is not proof of correct attribution. |
| `unavailable`, empty `overlaps` | The backend did not expose the required regular track. Overlap was not measured. |
| `assigned` | A model label exists and the span is not flagged by detected overlap. Not human-confirmed. |
| `provisional` | At least one detected overlap intersects the timed words, or the conservative fallback segment. `speaker` remains the proposed label, possibly null. |
| `unassigned` | No label exists and no detected overlap intersects this span. |

Intervals are in seconds from the start of the input recording, half-open
`[start, end)`, ordered, disjoint, and positive-duration. IDs are unique only
within this result. Preserve precision and namespace IDs by file/job when
combining recordings. `detected_speaker_count` is the number of distinct model
clusters active during the interval; it is neither the meeting's participant
count nor a list of confirmed speakers. Adjacent intervals with different active
cluster sets remain separate even when they have the same count.

## Attribution and rendering

Vemsa obtains the existing attribution track (exclusive by default) and the
regular overlap-capable track from **one** pyannote inference. It detects overlap
before caller labels can collapse several clusters into one name. It applies
review metadata **after** the existing attribution and smoothing steps.

Any positive-duration intersection flags the entire timed word conservatively;
a word touching only an interval boundary is not flagged. A segment is split at
word boundaries when its overlap references change. Word text, timestamps,
probabilities, and proposed speaker assignments are preserved. Original segment
text supplies punctuation. Missing word timings or text that cannot be anchored
reliably use a conservative whole-segment flag instead of fabricated word timing.
Overlap intervals remain in the result even when ASR supplied no words there.

Provisional text renders as:

```text
[00:00:01 - 00:00:01] [Överlappande tal – osäker talare]: ses
```

The existing text renderer displays whole seconds; the JSON above retains precise
timing. Spoken segment text never contains the marker. Downstream summaries and
exports must retain uncertainty until a human explicitly resolves attribution.
Readable words are retained. Vemsa does not declare them unintelligible, invent
missing words, duplicate a sentence under multiple speakers, or separate voices.

Custom/older diarizers without review support return `unavailable` when opted in.
`VEMSA_ENGINE=fake` emits a deterministic synthetic interruption solely for UI
smoke tests; its metadata is not audio analysis.

## Validation and rollout

Executable examples: [`tests/fixtures/speaker_review.json`](../tests/fixtures/speaker_review.json).
Each case contains input, review evidence, and expected output. Cases cover legacy,
clear, two/three voices, unavailable detection, and wordless unknown attribution.
`tests/test_speaker_review.py` validates them and covers all production engine
paths, JSON/multipart admission, the worker queue, and fresh PostgreSQL readback.

Model logs add only aggregate overlap duration, interval count, availability, and
provisional word count under `diarize.review`; no transcript, audio, embeddings,
or new confidence values are logged.

1. Ship this additive service contract with consumers still opted out.
2. Implement Eneo preservation, correction storage, rendering, and propagation.
3. Implement Lyssna and Eneo's built-in transcript review UI; run the shared cases
   through loading, editing, saving, approval, export, and reload.
4. Replay fixed synthetic WAVs through the deployed Vemsa alignment path, then
   validate on consented representative recordings. Inspect false overlap flags,
   missed overlap, latency, output size, and speaker-count sensitivity.
5. Enable the consumer request flag for a limited rollout. Roll back by stopping
   new opt-in requests while retaining the ability to read already stored results.

No clustering defaults or automatic centroid merging change in this release.
Synthetic checks establish correctness of the mechanics; they do not establish
real-meeting accuracy. CPU/GPU inference need not be bit-identical.

Related: [Eneo handover](handover-eneo-speaker-review.md),
[Lyssna handover](handover-lyssna-speaker-review.md),
[benchmark and replay procedure](speaker-review-evaluation.md).
