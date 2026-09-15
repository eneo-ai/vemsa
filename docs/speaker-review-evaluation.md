# Evaluate overlap and speaker review

The previous synthetic table is exploratory evidence, not a validated accuracy
baseline. Its pairwise score counted simultaneous correct reference speakers as
errors and used independent many-to-one label mapping, which could hide cluster
fragmentation. A perfect two-voice overlap could therefore look wrong. It also
omitted missed speech and false alarms. Neither “raw is worse than exclusive” nor
a specific cause for Ronny's recording follows from those numbers.

## What the corrected harness measures

`scripts/diarization_metrics.py` uses
[pyannote's DiarizationErrorRate](https://pyannote.github.io/pyannote-metrics/reference.html#diarization)
with optimal one-to-one label mapping, zero collar, and an explicit full-recording
window. It reports reference speaker-seconds, missed speech, false alarms,
confusion, and DER, both including and excluding reference overlap. A zero
reference denominator yields null DER. Exclusive output necessarily misses the
secondary voice during overlap; that is not automatically wrong-person confusion.

Overlap precision and recall measure wall-clock time with at least two distinct
voices, with null for an undefined denominator. Three simultaneous voices count
once; duplicate tracks of one identity do not constitute overlap. Pairwise cluster
occupancy remains available solely as a diagnostic matrix.

`scripts/synth_meeting.py` generator version 2:

- Separates random streams by feature and utterance. Adding backchannels, noise,
  or room effects does not change primary-turn scheduling. Drift deliberately
  changes utterance durations, so its timing is not an identical waveform ablation.
- Places main turns before backchannels, rejects self-overlap, and aligns placement
  to audio samples. A backchannel cannot overlap that same speaker's earlier or
  future turn. Empty TTS output fails immediately instead of becoming false truth.
- Writes versioned WAV, reference JSON, and caller-transcript JSON, with voice
  definitions, scenario, and the WAV's SHA-256. Reference intervals are trimmed TTS
  utterance windows, not manually verified speech activity or word-level truth.
- Evaluates the written PCM rather than the pre-quantized float buffer. Frozen WAVs
  can be replayed without regenerating voices. Evaluation artifacts retain both
  tracks, model name, installed versions, parameters, bounds, reference, timestamp,
  and audio hash, under a unique filename.

The four TTS voices, Nordic/Finnish accents reading Swedish, repeated script, and
simple acoustic effects are useful reproducible stressors, but not representative
natural conversation. The word-attribution simulation uses evenly spaced words:
its label count is explicitly a simulation, not production forced-alignment output.
Centroid cosine distances are diagnostic, not directly comparable to a clustering
threshold using a different distance/normalization. They do not justify automatic
merging. No clustering defaults change as part of this implementation.

The Community-1 model provides regular and exclusive diarization from a single
run; the exclusive track simplifies transcript reconciliation but does not prove
which voice produced a transcribed word. See the
[official model card](https://huggingface.co/pyannote/speaker-diarization-community-1).

## Repeatable runs

Install the `evaluation` group for the scoring tests. CI installs it alongside
`dev`; it does not require GPU inference or macOS voices. Audio generation needs
macOS `say` and the four voices declared in the script. Direct model evaluation
also needs the diarization extra and an accepted model license/token.

```sh
uv run --group evaluation pytest tests/test_synth_meeting.py
uv run --group evaluation python scripts/synth_meeting.py
uv run --group evaluation python scripts/synth_meeting.py --all --repeats 3
```

The implementation run generated this new fixed stress case:
`data/synth/meeting_v2_crosstalk+backchannels+room+moving+noise+drift_x3_s7.wav`,
with adjacent `.truth.json` and `.transcript.json`. It is approximately 9.4 minutes
with four synthetic identities and 86.1 seconds of reference overlap. Generated
media and evaluation reports remain under ignored `data/synth`, not in git.

Use the *same* audio/truth pair for unbounded versus `--max-speakers 4` versus
`--num-speakers 4`. Do not regenerate the audio between these comparisons:

```sh
uv run --group evaluation --extra diarize --extra cpu python scripts/synth_meeting.py \
  --audio 'data/synth/meeting_v2_crosstalk+backchannels+room+moving+noise+drift_x3_s7.wav' \
  --truth 'data/synth/meeting_v2_crosstalk+backchannels+room+moving+noise+drift_x3_s7.truth.json' \
  --eval
```

This command is a direct CPU/in-memory model harness, not the Vemsa HTTP path.
GPU results need separate measurement; the same pipeline need not produce
bit-identical CPU/GPU clusters. The Mac's known torchcodec decoding problem is
bypassed by this harness, so this result cannot validate service decoding.

## Replay the actual service, including alignment

Use `scripts/replay_synth_meeting.py` against an explicitly selected test Vemsa
service with its real alignment stack. Set `VEMSA_EVAL_TOKEN` in the environment;
the script never saves it. Only generator-v2 fixtures whose WAV hash matches their
truth file are accepted. This submits the synthetic audio to that chosen service.

```sh
uv run python scripts/replay_synth_meeting.py --base-url http://localhost:8000 \
  --audio 'data/synth/meeting_v2_crosstalk+backchannels+room+moving+noise+drift_x3_s7.wav' \
  --truth 'data/synth/meeting_v2_crosstalk+backchannels+room+moving+noise+drift_x3_s7.truth.json' \
  --task diarize --max-speakers 4
```

Replace the example URL with the approved test service. `task=diarize` sends the
primary-turn caller transcript and exercises real forced alignment when
`VEMSA_DIARIZE_PREFER_ALIGN=true`; omitted backchannels are intentional. Confirm
`alignment=forced` in the saved result. `--task transcribe` also exercises ASR;
`--legacy` compares with review opted out. Jobs/results are saved as
`<audio stem>.<job id>.service.json`, with the exact request and audio hash. A
client timeout leaves a retrievable job, not a claimed successful result.

Before enabling consumer opt-in, replay with and without review, with unbounded
and known-count priors. Check preserved words, proposed labels, references,
rendering, alignment rung, latency, and payload growth. Measure overlap detection
against reference windows separately from transcription/attribution correctness.
A correct speaker count alone is not sufficient.

## Recorded CPU smoke result (14 September 2026)

The version-2, seed-7, three-repeat stress fixture was evaluated with Community-1
and default clustering (`threshold=0.6`, `Fa=0.07`, `Fb=0.8`), without a speaker
count prior. It produced **6 clusters for 4 reference identities**. With the
corrected scorer, raw DER was **26.65%** and exclusive DER was **31.07%** including
overlap; excluding reference overlap they were **20.71%** and **19.82%**. These
track scores measure different capabilities and are not word-attribution accuracy.

Overlap detection achieved **92.65% precision** and **63.81% recall** against the
TTS reference windows: 54.97 matched seconds, 59.33 detected seconds, and 86.14
reference seconds. This is one synthetic case, not a calibrated reliability
estimate for natural meetings. In particular, the missed overlap supports keeping
absence of a flag distinct from a claim of certainty.

Audio SHA-256:
`1df663e894e5bac6da29787d2434f6118e2d453c649c72eb440d5fbd2d71acd5`.
The full `.20260914T125248220130Z.eval.json` artifact is beside the WAV. Installed
versions were `pyannote.audio=4.0.3`, `pyannote.metrics=4.1`, `torch=2.8.0`, `numpy=2.4.6`.
No count-prior comparison or real service/GPU accuracy result is claimed by this
single run.

## Remaining validation

GPU/service replay and representative consented recordings remain release checks.
Use speakers with similar voices, short interjections, unequal speaking time,
laughter, movement, and natural simultaneous speech. Record per-speaker channels
when feasible to make reference annotation easier; evaluate the mixed audio.
Have a reviewer identify the words and attribution they can actually hear, and
record unresolved regions explicitly. Freeze a held-out set before any tuning.
Ronny's non-public recording is not required or requested.

Until these checks exist, do not promise a reduction from nine labels to four,
select a centroid merge threshold, change VBx defaults, infer a posterior
confidence from turn purity, or treat the detector's absence of overlap as proof
of reliable attribution.
