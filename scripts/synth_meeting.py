# ruff: noqa: E501
"""Synthetic multi-speaker meeting for diarization checks.

Builds a Swedish four-person meeting from macOS ``say`` voices, with the
confounders a real recording has and a text-to-speech clip does not — crosstalk,
backchannels, per-speaker mic distance and room reverb, a speaker who moves, a
noise floor, and intra-speaker pitch/rate drift — each switchable so a wrong
speaker count can be bisected to its cause. Writes the clip, the ground truth,
and a caller transcript shaped like what eneo submits for ``task=diarize``, then
(``--eval``) runs the same pyannote pipeline Vemsa uses and reports detected
versus true speakers and overlap-aware DER. The optional word attribution
simulation uses uniformly spaced words, not the service's forced alignment.

    uv run python scripts/synth_meeting.py --eval                      # clean baseline
    uv run python scripts/synth_meeting.py --eval --all                # every confounder
    uv run python scripts/synth_meeting.py --eval --all --max-speakers 4
    uv run python scripts/synth_meeting.py --eval --all --threshold 0.7 --fb 4

Needs macOS ``say`` (the Swedish voice Alva; the other three voices are Nordic
and Finnish system voices reading Swedish; not representative natural speech) and the
``diarize`` extra with HF_TOKEN (``.env`` is read). Clips are cached under
``data/synth/tts`` so re-runs only mix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import warnings
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np

GENERATOR_VERSION = 2
SR = 16000
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "synth"
TTS_CACHE = OUT_DIR / "tts"


@dataclass(frozen=True)
class Voice:
    name: str  # ground-truth speaker label
    say_voice: str
    gain: float  # mic distance: 1.0 close, 0.3 far
    lowpass_hz: float | None  # far/turned-away speakers lose highs
    reverb: float  # wet level of the room impulse
    base_rate: int = 180


# Two women, two men; Alva is the only Swedish voice, the others are Nordic and
# Finnish system voices reading Swedish text. The ``room`` confounder applies
# gain/lowpass/reverb; without it every speaker is close-miked and dry.
VOICES = [
    Voice("ANNA", "Alva", 1.0, None, 0.10, 175),
    Voice("NORA", "Nora", 0.75, None, 0.18, 185),
    Voice("EDDY", "Eddy (Finnish (Finland))", 0.45, 4000.0, 0.30, 170),
    Voice("GÖRAN", "Grandpa (Finnish (Finland))", 0.35, 3200.0, 0.35, 165),
]

# A planning meeting. "!" after the speaker marks an interruption: the line
# starts while the previous one is still running (fully simultaneous for a
# stretch) when crosstalk is enabled.
SCRIPT = [
    (
        "ANNA",
        "Hej allihopa och välkomna. Vi har tre punkter idag: cykelvägen längs ån, budgeten för nästa kvartal och rekryteringen av en ny projektledare.",
    ),
    (
        "NORA",
        "Tack Anna. Kan vi börja med cykelvägen? Jag har fått flera frågor från boende på Strandvägen om när grävningen börjar.",
    ),
    (
        "EDDY",
        "Entreprenören säger vecka fjorton, men det förutsätter att bygglovet vinner laga kraft innan dess.",
    ),
    (
        "GÖRAN",
        "Och det gör det inte om grannarna överklagar igen. Jag pratade med en av dem i förra veckan och han var inte nöjd.",
    ),
    ("ANNA", "Vad var det han var missnöjd med, konkret?"),
    (
        "GÖRAN",
        "Framför allt belysningen. Han tycker att stolparna kommer lysa rakt in i sovrummet.",
    ),
    (
        "NORA!",
        "Men det där har vi ju redan löst, vi bytte till lägre stolpar med avskärmning i februari.",
    ),
    ("GÖRAN", "Det vet inte han. Ingen har berättat det för honom."),
    (
        "EDDY",
        "Då föreslår jag att vi skickar ut ett informationsbrev till alla längs sträckan innan vi gör något annat.",
    ),
    (
        "ANNA",
        "Bra. Nora, kan du ta det? Ett brev på högst en sida, med en karta och de nya stolparna inritade.",
    ),
    ("NORA", "Absolut, jag har ett utkast klart på torsdag."),
    ("ANNA", "Då går vi vidare till budgeten. Eddy, du har siffrorna."),
    (
        "EDDY",
        "Ja. Vi ligger på fyrtiosju procent av årsbudgeten efter första kvartalet, vilket är ungefär tolv procent över plan.",
    ),
    ("GÖRAN!", "Tolv procent? Det är ju mycket mer än vi sa i januari."),
    (
        "EDDY",
        "Det beror nästan helt på asfaltpriserna. De gick upp med tjugo procent i mars och vi hade inte hunnit binda priset.",
    ),
    ("NORA", "Kan vi flytta något till nästa år för att kompensera?"),
    (
        "EDDY",
        "Den nya bron vid Kvarnbacken går att skjuta på. Den är budgeterad till fyra komma två miljoner.",
    ),
    ("ANNA", "Jag är tveksam. Bron var ett vallöfte och vi har redan flyttat den en gång."),
    (
        "GÖRAN",
        "Jag håller med Anna. Skjut hellre på parkeringen vid idrottsplatsen, den är ingen som frågar efter.",
    ),
    ("NORA!", "Fast idrottsföreningen frågar efter den varje månad, Göran."),
    ("GÖRAN", "Jo, men de kan vänta ett halvår till."),
    (
        "ANNA",
        "Vi tar det till nämnden. Eddy, gör två alternativ: ett där bron skjuts och ett där parkeringen skjuts, med konsekvenser för varje.",
    ),
    ("EDDY", "Ska bli. Jag behöver en vecka."),
    (
        "ANNA",
        "Sista punkten. Rekryteringen. Vi har fått in tjugotre ansökningar och kallat fem till intervju.",
    ),
    (
        "NORA",
        "Jag satt med på tre av intervjuerna. Två av dem var riktigt starka, särskilt hon som kommer från Trafikverket.",
    ),
    (
        "EDDY",
        "Hon hade väldigt bra svar på frågan om konflikter med entreprenörer, det tyckte jag också.",
    ),
    ("GÖRAN", "Vad hade hon för löneanspråk?"),
    ("NORA", "Femtiotvå tusen. Det är inom spannet men i övre delen."),
    ("GÖRAN", "Det är mer än vad jag har."),
    ("ANNA!", "Det får vi ta i ett annat möte, Göran."),
    ("EDDY", "Jag tycker vi går vidare med henne och tar referenser den här veckan."),
    ("NORA", "Jag ringer referenserna imorgon förmiddag."),
    ("ANNA", "Bra. Något övrigt?"),
    (
        "GÖRAN",
        "En sak. Kaffemaskinen på tredje våningen är trasig igen. Det är tredje gången i år.",
    ),
    ("NORA", "Jag har redan felanmält den, men de säger att reservdelen kommer först i maj."),
    ("EDDY!", "Maj? Då köper vi en ny, det kostar mindre än vad vi förlorar i arbetstid."),
    (
        "ANNA",
        "Eddy, kolla vad en ny kostar och skicka till mig. Om det är under tio tusen tar jag det på min budget.",
    ),
    ("EDDY", "Det är det garanterat."),
    (
        "ANNA",
        "Då tackar jag för idag. Nästa möte är om två veckor, samma tid. Nora skickar anteckningarna.",
    ),
    ("NORA", "Det gör jag. Tack allihopa."),
    ("GÖRAN", "Tack, hej då."),
    ("EDDY", "Hej då."),
]

BACKCHANNELS = ["Mm.", "Ja.", "Precis.", "Okej.", "Absolut.", "Just det.", "Mm, ja."]


@dataclass
class Placed:
    speaker: str
    start: float
    end: float
    text: str
    kind: str  # turn | backchannel

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Scenario:
    seed: int = 7
    crosstalk: bool = False  # turn changes overlap; "!" lines interrupt
    backchannels: bool = False  # short "mm"/"ja" from others during long turns
    room: bool = False  # per-speaker distance, lowpass, reverb
    moving: bool = False  # one speaker walks away halfway (needs room)
    noise: bool = False  # pink noise floor and a hum
    drift: bool = False  # per-utterance pitch and rate variation
    repeats: int = 1  # replay the script this many times (longer meeting)
    knobs: dict = field(default_factory=dict)

    def slug(self) -> str:
        on = [
            k
            for k in ("crosstalk", "backchannels", "room", "moving", "noise", "drift")
            if getattr(self, k)
        ]
        base = "+".join(on) or "clean"
        return f"meeting_v{GENERATOR_VERSION}_{base}_x{self.repeats}_s{self.seed}"


def synth(voice: Voice, text: str, *, rate: int, pitch: int) -> np.ndarray:
    """One utterance as float32 mono at 16 kHz, cached by content."""
    import soundfile

    TTS_CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{voice.say_voice}|{rate}|{pitch}|{text}".encode()).hexdigest()[:16]
    path = TTS_CACHE / f"{key}.wav"
    if not path.exists() or soundfile.info(path).frames == 0:
        spoken = f"[[pbas {pitch:+d}]] {text}" if pitch else text
        subprocess.run(
            [
                "say",
                "-v",
                voice.say_voice,
                "-r",
                str(rate),
                "--file-format=WAVE",
                f"--data-format=LEI16@{SR}",
                "-o",
                str(path),
                spoken,
            ],
            check=True,
        )
    data, sr = soundfile.read(path, dtype="float32")
    if not data.size:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            "macOS say produced empty audio; check voice availability and local speech-service access"
        )
    assert sr == SR, sr
    if data.ndim > 1:
        data = data.mean(axis=1)
    return trim_silence(data)


def trim_silence(data: np.ndarray, thresh: float = 0.005, pad_s: float = 0.05) -> np.ndarray:
    loud = np.flatnonzero(np.abs(data) > thresh)
    if loud.size == 0:
        return data
    pad = int(pad_s * SR)
    return data[max(0, loud[0] - pad) : min(len(data), loud[-1] + pad)]


def room_ir(rng: np.random.Generator, rt_s: float = 0.35) -> np.ndarray:
    n = int(rt_s * 2 * SR)
    t = np.arange(n) / SR
    return (rng.standard_normal(n) * np.exp(-t / (rt_s / 3))).astype(np.float32) / 40.0


def lowpass(data: np.ndarray, cutoff_hz: float) -> np.ndarray:
    from scipy.signal import butter, lfilter

    b, a = butter(4, cutoff_hz / (SR / 2))
    return lfilter(b, a, data).astype(np.float32)


def pink_noise(rng: np.random.Generator, n: int) -> np.ndarray:
    from scipy.signal import lfilter

    white = rng.standard_normal(n)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    pink = lfilter(b, a, white)
    return (pink / (np.abs(pink).max() + 1e-9)).astype(np.float32)


def build(scn: Scenario) -> tuple[np.ndarray, list[Placed]]:
    if scn.repeats < 1 or (scn.moving and not scn.room):
        raise ValueError("repeats must be positive and moving requires room")
    # Feature-local streams prevent adding backchannels/room/noise from changing
    # later turn decisions. Drift changes utterance durations intentionally.
    nrng = np.random.default_rng(scn.seed)
    ir_rng = np.random.default_rng(scn.seed + 101)
    voices = {v.name: v for v in VOICES}
    ir = room_ir(ir_rng)

    lines = SCRIPT * scn.repeats
    placed: list[Placed] = []
    tracks: list[
        tuple[float, np.ndarray, Voice, float]
    ] = []  # start_s, audio, voice, position 0..1
    cursor = 0.0
    prev_end = 0.0
    total_lines = len(lines)
    speaker_ends: dict[str, float] = {}
    for index, (tag, text) in enumerate(lines):
        rng = random.Random(f"{scn.seed}:timing:{index}")
        drift_rng = random.Random(f"{scn.seed}:drift:{index}")
        name = tag.rstrip("!")
        interrupt = tag.endswith("!") and scn.crosstalk
        voice = voices[name]
        rate = voice.base_rate
        pitch = 0
        if scn.drift:
            rate = voice.base_rate + drift_rng.randint(-25, 25)
            pitch = drift_rng.randint(-6, 6)
        audio = synth(voice, text, rate=rate, pitch=pitch)
        dur = len(audio) / SR
        if index == 0:
            start = 0.5
        elif interrupt:
            # barge in: start 1.5–2.5 s before the previous line ends
            start = max(prev_end - rng.uniform(1.5, 2.5), cursor)
        elif scn.crosstalk and rng.random() < 0.5:
            start = prev_end - rng.uniform(0.2, 0.9)  # eager turn-taking
        else:
            start = prev_end + rng.uniform(0.25, 0.9)
        start = max(start, 0.0, speaker_ends.get(name, 0.0), cursor)
        start = round(start * SR) / SR
        end = start + dur
        speaker_ends[name] = end
        placed.append(Placed(name, start, end, text, "turn"))
        tracks.append((start, audio, voice, index / max(1, total_lines - 1)))
        prev_end = end
        cursor = start

    # Main turns are fixed first: a backchannel must not collide with that same
    # voice's previous OR future turn, or another of its accepted backchannels.
    if scn.backchannels:
        for index, host in enumerate(list(placed)):
            rng = random.Random(f"{scn.seed}:backchannel:{index}")
            dur = host.end - host.start
            if dur <= 4.0 or rng.random() >= 0.6:
                continue
            other = rng.choice([v for v in VOICES if v.name != host.speaker])
            word = rng.choice(BACKCHANNELS)
            bc = synth(other, word, rate=other.base_rate, pitch=0)
            at = round((host.start + rng.uniform(1.0, dur - 1.0)) * SR) / SR
            end = at + len(bc) / SR
            if end > host.end or any(
                p.speaker == other.name and at < p.end and end > p.start for p in placed
            ):
                continue
            placed.append(Placed(other.name, at, end, word, "backchannel"))
            tracks.append((at, bc, other, index / max(1, total_lines - 1)))
    validate_truth(placed)

    length = int((max(p.end for p in placed) + 1.0) * SR)
    mix = np.zeros(length, dtype=np.float32)
    for start, audio, voice, position in tracks:
        sig = audio
        if scn.room:
            gain, cutoff, wet = voice.gain, voice.lowpass_hz, voice.reverb
            if scn.moving and voice.name == "NORA" and position > 0.5:
                gain, cutoff, wet = 0.3, 2800.0, 0.45  # walked to the whiteboard
            if cutoff:
                sig = lowpass(sig, cutoff)
            dry = sig * gain
            from scipy.signal import fftconvolve

            wet_sig = fftconvolve(sig, ir)[: len(sig)] * gain * wet
            sig = (dry + wet_sig).astype(np.float32)
        i = round(start * SR)
        mix[i : i + len(sig)] += sig[: max(0, length - i)]
    if scn.noise:
        mix += pink_noise(nrng, length) * 0.01
        t = np.arange(length) / SR
        mix += (0.002 * np.sin(2 * np.pi * 50 * t) + 0.001 * np.sin(2 * np.pi * 100 * t)).astype(
            np.float32
        )
    peak = np.abs(mix).max()
    if peak > 0.95:
        mix *= 0.95 / peak
    placed.sort(key=lambda p: p.start)
    return mix, placed


def validate_truth(placed: list[Placed]) -> None:
    ends: dict[str, float] = {}
    for p in sorted(placed, key=lambda p: p.start):
        if not (np.isfinite(p.start) and np.isfinite(p.end) and 0 <= p.start < p.end):
            raise ValueError("invalid reference timestamps")
        if p.start < ends.get(p.speaker, 0) - 1e-9:
            raise ValueError(f"impossible self-overlap for {p.speaker}")
        ends[p.speaker] = p.end


def write_outputs(scn: Scenario, mix: np.ndarray, placed: list[Placed]) -> Path:
    import soundfile

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / scn.slug()
    wav = base.with_suffix(".wav")
    soundfile.write(wav, mix, SR, subtype="PCM_16")
    base.with_suffix(".truth.json").write_text(
        json.dumps(
            {
                "generator_version": GENERATOR_VERSION,
                "scenario": asdict(scn),
                "voices": [asdict(v) for v in VOICES],
                "audio_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
                "reference_kind": "TTS utterance windows; includes residual silence, not word truth",
                "segments": [p.as_dict() for p in placed],
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    # what eneo submits for task=diarize: text with windows, no speakers, no backchannels
    base.with_suffix(".transcript.json").write_text(
        json.dumps(
            {
                "segments": [
                    {"text": p.text, "start": p.start, "end": p.end}
                    for p in placed
                    if p.kind == "turn"
                ]
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    return wav


# --- evaluation -------------------------------------------------------------


def load_pipeline(clustering_overrides: dict[str, float]):
    warnings.filterwarnings("ignore")
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))
    from pyannote.audio import Pipeline

    model = os.environ.get("VEMSA_DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1")
    pipe = Pipeline.from_pretrained(
        model,
        token=os.environ.get("VEMSA_HF_TOKEN") or os.environ.get("HF_TOKEN"),
        cache_dir=str(ROOT / "data" / "models"),
    )
    if pipe is None:
        sys.exit(f"could not load {model}: accept its license on Hugging Face and set HF_TOKEN")
    if clustering_overrides:
        params = pipe.parameters(instantiated=True)
        params["clustering"].update(clustering_overrides)
        pipe.instantiate(params)
    print("pipeline", model, pipe.parameters(instantiated=True)["clustering"])
    return pipe


def diarize(pipe, mix: np.ndarray, bounds: dict[str, int]):
    import torch

    # Direct in-memory CPU harness; excludes service decoding, ASR and alignment.
    # GPU output and actual Vemsa attribution require separate validation.
    return pipe({"waveform": torch.from_numpy(mix)[None], "sample_rate": SR}, **bounds)


def overlap_matrix(
    hyp: list[tuple[float, float, str]], truth: list[Placed]
) -> dict[str, dict[str, float]]:
    matrix: dict[str, dict[str, float]] = {}
    for hs, he, hl in hyp:
        row = matrix.setdefault(hl, {})
        for p in truth:
            o = max(0.0, min(he, p.end) - max(hs, p.start))
            if o > 0:
                row[p.speaker] = row.get(p.speaker, 0.0) + o
    return matrix


def report(
    name: str, hyp: list[tuple[float, float, str]], truth: list[Placed], duration: float
) -> dict:
    from diarization_metrics import score_diarization

    reference = [(p.start, p.end, p.speaker) for p in truth]
    scores = {
        "clusters": len({label for _, _, label in hyp}),
        "reference_speakers": len({p.speaker for p in truth}),
        "including_overlap": score_diarization(reference, hyp, duration=duration),
        "excluding_reference_overlap": score_diarization(
            reference, hyp, duration=duration, skip_overlap=True
        ),
        # Pairwise occupancy is diagnostic only; it is not an assignment/error score.
        "pairwise_overlap_seconds": overlap_matrix(hyp, truth),
    }
    print(f"\n[{name}] {scores['clusters']} clusters, truth {scores['reference_speakers']}")
    for key in ("including_overlap", "excluding_reference_overlap"):
        print(f"  {key}: {json.dumps(scores[key])}")
    return scores


def vemsa_attribution(
    hyp: list[tuple[float, float, str]], truth: list[Placed], bounds: dict[str, int]
) -> None:
    """How many speakers the rendered transcript would show: eneo's segments,
    words spread evenly over each window (a stand-in for forced alignment), then
    Vemsa's own word→speaker attribution and smoothing."""
    sys.path.insert(0, str(ROOT / "src"))
    from vemsa.jobs.models import Word
    from vemsa.pipeline.diarize import Turn, assign_speakers

    words: list[Word] = []
    for p in truth:
        if p.kind != "turn":
            continue
        tokens = p.text.split()
        step = (p.end - p.start) / len(tokens)
        for i, tok in enumerate(tokens):
            words.append(
                Word(
                    word=tok,
                    start=round(p.start + i * step, 3),
                    end=round(p.start + (i + 1) * step, 3),
                    probability=1.0,
                )
            )
    turns = [Turn(start=s, end=e, speaker=lab) for s, e, lab in hyp]
    segments = assign_speakers(words, turns)
    labels = [seg.speaker for seg in segments]
    counts = {lab: labels.count(lab) for lab in sorted(set(labels), key=str)}
    print(
        f"  SIMULATION (uniform word times, not forced alignment): {len(counts)} labels over {len(segments)} segments  {counts}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for flag in ("crosstalk", "backchannels", "room", "moving", "noise", "drift"):
        ap.add_argument(f"--{flag}", action="store_true")
    ap.add_argument("--all", action="store_true", help="every confounder on")
    ap.add_argument(
        "--repeats", type=int, default=1, help="replay the script N times (longer meeting)"
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eval", action="store_true", help="run pyannote and report")
    ap.add_argument("--num-speakers", type=int)
    ap.add_argument("--min-speakers", type=int)
    ap.add_argument("--max-speakers", type=int)
    ap.add_argument("--threshold", type=float, help="clustering.threshold override")
    ap.add_argument("--fa", type=float, help="clustering.Fa override (VBx)")
    ap.add_argument("--fb", type=float, help="clustering.Fb override (VBx)")
    ap.add_argument(
        "--audio", type=Path, help="evaluate an existing frozen WAV; no TTS regeneration"
    )
    ap.add_argument("--truth", type=Path, help="reference JSON paired with --audio")
    args = ap.parse_args()
    if bool(args.audio) != bool(args.truth):
        ap.error("--audio and --truth must be supplied together")
    if args.repeats < 1 or (args.moving and not (args.room or args.all)):
        ap.error("repeats must be positive and moving requires room")

    scn = Scenario(
        seed=args.seed,
        crosstalk=args.crosstalk or args.all,
        backchannels=args.backchannels or args.all,
        room=args.room or args.all,
        moving=args.moving or args.all,
        noise=args.noise or args.all,
        drift=args.drift or args.all,
        repeats=args.repeats,
    )
    if args.audio:
        import soundfile

        wav = args.audio
        truth_data = json.loads(args.truth.read_text())
        expected_hash = truth_data.get("audio_sha256")
        if expected_hash and hashlib.sha256(wav.read_bytes()).hexdigest() != expected_hash:
            ap.error("audio does not match the reference hash")
        mix, sr = soundfile.read(wav, dtype="float32")
        if sr != SR or mix.ndim != 1:
            ap.error("frozen audio must be mono 16 kHz")
        placed = [Placed(**p) for p in truth_data["segments"]]
        validate_truth(placed)
    else:
        mix, placed = build(scn)
        wav = write_outputs(scn, mix, placed)
        # Evaluate the actual PCM file, not the unquantized pre-write buffer.
        import soundfile

        mix, _ = soundfile.read(wav, dtype="float32")
        truth_data = json.loads(wav.with_suffix(".truth.json").read_text())
    from diarization_metrics import overlap_intervals, overlap_scores

    reference = [(p.start, p.end, p.speaker) for p in placed]
    overlapped = sum(e - s for s, e in overlap_intervals(reference))
    print(
        f"audio {wav} ({len(mix) / SR / 60:.1f} min; {overlapped:.1f} wall-clock seconds overlapping)"
    )
    if not args.eval:
        return

    overrides = {
        k: v
        for k, v in (("threshold", args.threshold), ("Fa", args.fa), ("Fb", args.fb))
        if v is not None
    }
    bounds = {
        k: v
        for k, v in (
            ("num_speakers", args.num_speakers),
            ("min_speakers", args.min_speakers),
            ("max_speakers", args.max_speakers),
        )
        if v is not None
    }
    from vemsa.jobs.models import JobRequest

    JobRequest(**bounds)  # same count-prior validation as service admission
    pipe = load_pipeline(overrides)
    out = diarize(pipe, mix, bounds)
    raw = [
        (s.start, s.end, str(lab))
        for s, _, lab in out.speaker_diarization.itertracks(yield_label=True)
    ]
    exclusive = [
        (s.start, s.end, str(lab))
        for s, _, lab in out.exclusive_speaker_diarization.itertracks(yield_label=True)
    ]
    print(f"bounds {bounds or 'none'}")
    duration = len(mix) / SR
    scores = {
        "raw": report("raw diarization", raw, placed, duration),
        "exclusive": report(
            "exclusive (missing secondary speech counts as misses)", exclusive, placed, duration
        ),
        "overlap_detection": overlap_scores(reference, raw),
    }
    artifact = {
        "evaluation_version": 2,
        "harness": "direct pyannote CPU / decoded PCM; no service ASR or forced alignment",
        "created_at": datetime.now(UTC).isoformat(),
        "audio_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
        "truth": truth_data,
        "model": os.environ.get(
            "VEMSA_DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1"
        ),
        "parameters": pipe.parameters(instantiated=True),
        "bounds": bounds,
        "versions": {
            package: version(package)
            for package in ("pyannote.audio", "pyannote.metrics", "torch", "numpy")
        },
        "scores": scores,
        "raw_turns": raw,
        "exclusive_turns": exclusive,
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    evaluation_path = wav.with_suffix(f".{stamp}.eval.json")
    evaluation_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    print(f"saved {evaluation_path}")
    print(f"overlap detection: {scores['overlap_detection']}")
    vemsa_attribution(exclusive, placed, bounds)
    if out.speaker_embeddings is not None:
        emb = out.speaker_embeddings / np.linalg.norm(out.speaker_embeddings, axis=1, keepdims=True)
        labels = out.speaker_diarization.labels()
        dist = 1 - emb @ emb.T
        print(
            "\n  cosine distance between cluster centroids (diagnostic only; not the clustering threshold scale)",
        )
        print("  " + " " * 12 + "".join(f"{lab:>12}" for lab in labels))
        for i, lab in enumerate(labels):
            print(f"  {lab:<12}" + "".join(f"{dist[i, j]:>12.2f}" for j in range(len(labels))))


if __name__ == "__main__":
    main()
