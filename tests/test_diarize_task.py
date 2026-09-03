"""task=diarize: speaker labels for an externally produced transcript."""

import asyncio
import contextlib
import json
import subprocess
from pathlib import Path

import httpx
import pytest
import respx
from httpx import Response

from conftest import FakeEngine
from vemsa.config import Settings
from vemsa.jobs.models import Segment, SpeakerBounds, Word
from vemsa.main import create_app
from vemsa.pipeline import align
from vemsa.pipeline.diarize import Turn, _decodable_audio
from vemsa.pipeline.diarize_only import DiarizeOnlyEngine
from vemsa.pipeline.fake import CannedEngine
from vemsa.pipeline.label import label_speakers, words_plausible
from vemsa.pipeline.whisper_api import OpenAIWhisperEngine

AUTH = {"Authorization": "Bearer secret-token"}
WORDS = [
    {"word": "hej", "start": 0.0, "end": 0.4},
    {"word": "och", "start": 0.5, "end": 0.7},
    {"word": "tack", "start": 2.0, "end": 2.2},
]


@contextlib.asynccontextmanager
async def api_client(settings: Settings, engine=None):
    app = create_app(settings=settings, engine=engine or FakeEngine())
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client, app


async def poll_until(client: httpx.AsyncClient, job_id: str, wanted: str, timeout: float = 5.0):
    async with asyncio.timeout(timeout):
        while True:
            response = await client.get(f"/v1/jobs/{job_id}", headers=AUTH)
            assert response.status_code == 200
            if response.json()["status"] == wanted:
                return response.json()
            await asyncio.sleep(0.01)


def multipart(**fields: str) -> dict:
    return {"files": {"file": ("a.wav", b"fake audio")}, "data": fields}


async def submit(client: httpx.AsyncClient, **fields: str) -> httpx.Response:
    return await client.post("/v1/jobs", headers=AUTH, **multipart(**fields))


async def test_multipart_diarize_lifecycle_echoes_external_model(settings: Settings):
    engine = FakeEngine()
    async with api_client(settings, engine) as (client, _):
        response = await submit(client, task="diarize", language="sv", words=json.dumps(WORDS))
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        await poll_until(client, job_id, "completed")

        result = (await client.get(f"/v1/jobs/{job_id}/result", headers=AUTH)).json()
        assert result["model"] == "external"

    call = engine.calls[-1]
    assert call["task"] == "diarize"
    assert [w.word for w in call["words"]] == ["hej", "och", "tack"]
    assert call["model"] == "external"
    assert call["language"] == "sv"


@respx.mock
async def test_json_submission_with_source_url(settings: Settings):
    respx.get("https://example.org/m.mp3").mock(return_value=Response(200, content=b"audio"))
    async with api_client(settings) as (client, _):
        response = await client.post(
            "/v1/jobs",
            json={
                "task": "diarize",
                "source_url": "https://example.org/m.mp3",
                "segments": [{"start": 0.0, "end": 1.4, "text": "hej och välkomna"}],
            },
            headers=AUTH,
        )
        assert response.status_code == 202
        await poll_until(client, response.json()["job_id"], "completed")


async def test_submission_validation_failures(settings: Settings):
    async with api_client(settings) as (client, _):
        # diarize without any transcript
        assert (await submit(client, task="diarize")).status_code == 422
        # transcribe with transcript fields
        assert (await submit(client, words=json.dumps(WORDS))).status_code == 422
        # diarize=false contradiction
        assert (
            await submit(client, task="diarize", words=json.dumps(WORDS), diarize="false")
        ).status_code == 422
        # malformed JSON part
        response = await submit(client, task="diarize", words="[{broken")
        assert response.status_code == 422
        assert "not valid JSON" in response.json()["detail"]
        # negative / inverted timestamps
        bad = [{"word": "x", "start": -1.0, "end": 0.5}]
        assert (await submit(client, task="diarize", words=json.dumps(bad))).status_code == 422
        inverted = [{"word": "x", "start": 2.0, "end": 1.0}]
        assert (await submit(client, task="diarize", words=json.dumps(inverted))).status_code == 422


async def test_oversized_transcript_is_413(settings: Settings):
    settings.max_transcript_bytes = 200
    async with api_client(settings) as (client, _):
        # over our cap but under the raised starlette part limit: our contract 413
        words = [{"word": f"w{i}", "start": float(i), "end": i + 0.5} for i in range(20)]
        response = await submit(client, task="diarize", words=json.dumps(words))
        assert response.status_code == 413
        assert "transcript exceeds" in response.json()["detail"]
        # big enough that starlette's part parser trips first: still a 413
        huge = "x" * (200 + 65536 + 1024)
        response = await submit(client, task="diarize", words=huge)
        assert response.status_code == 413


async def test_diarize_tier_refuses_transcribe_and_is_ready(settings: Settings):
    settings.engine = "diarize"
    async with api_client(settings, engine=None) as (client, _):
        ready = await client.get("/readyz")
        assert ready.status_code == 200 and ready.json()["status"] == "ready"
        response = await submit(client, language="sv")
        assert response.status_code == 422
        assert "task=diarize" in response.json()["detail"]


async def test_fake_engine_alternates_speakers(tmp_path: Path):
    engine = CannedEngine()
    result = engine.label_speakers(
        tmp_path / "a.wav",
        words=[],
        segments=[
            Segment(start=0.0, end=1.0, text="hej"),
            Segment(start=2.5, end=3.0, text="tack"),
        ],
        language="sv",
        model="external",
    )
    assert [s.speaker for s in result.segments] == ["SPEAKER_00", "SPEAKER_01"]
    assert result.model == "external"
    assert result.language == "sv"
    assert "SPEAKER_00" in result.text and "SPEAKER_01" in result.text


class FakeDiarizer:
    def __init__(self) -> None:
        self.calls: list[Path] = []
        self.speaker_bounds: list[object] = []

    def diarize(self, audio_path: Path, *, speakers=None) -> list[Turn]:
        self.calls.append(audio_path)
        self.speaker_bounds.append(speakers)
        return [Turn(0.0, 1.5, "SPEAKER_00"), Turn(1.9, 2.7, "SPEAKER_01")]

    def load(self) -> None:
        pass


def test_label_speakers_word_level_merge(tmp_path: Path):
    words = [Word(**w) for w in WORDS]
    result = label_speakers(
        FakeDiarizer(),
        tmp_path / "missing.wav",
        words=words,
        segments=[],
        language="auto",
        model="external",
    )
    assert [s.speaker for s in result.segments] == ["SPEAKER_00", "SPEAKER_01"]
    assert result.language == "unknown"
    # audio unreadable: duration falls back to the last labelled end
    assert result.duration_seconds == 2.2
    assert result.text.count("\n") == 1


def test_label_speakers_reuses_the_engine_diarizer(settings: Settings, tmp_path: Path):
    settings.whisper_api_base = "http://whisper.local/v1"
    diarizer = FakeDiarizer()
    engine = OpenAIWhisperEngine(settings, diarizer=diarizer)
    # the engine prefers alignment (words are set aside and realigned); stub the
    # aligner so the test runs without the ML stack
    engine._segment_aligner = fake_aligner
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake audio")
    engine.label_speakers(
        audio, words=[Word(**w) for w in WORDS], segments=[], language="sv", model="external"
    )
    assert diarizer.calls == [audio]


class SwappedDiarizer(FakeDiarizer):
    """The same turns as FakeDiarizer, numbered the other way round — what a
    fresh pyannote run does to a transcript a human already labelled."""

    def diarize(self, audio_path: Path, *, speakers=None) -> list[Turn]:
        super().diarize(audio_path, speakers=speakers)
        return [Turn(0.0, 1.5, "SPEAKER_01"), Turn(1.9, 2.7, "SPEAKER_00")]


def test_caller_speaker_labels_survive_a_rerun(tmp_path: Path):
    # the caller's labelled segments (from an earlier run, possibly renamed by a
    # human) are the reference the new clusters are mapped onto
    result = label_speakers(
        SwappedDiarizer(),
        tmp_path / "missing.wav",
        words=[Word(**w) for w in WORDS],
        segments=[
            Segment(start=0.0, end=0.7, speaker="Anna", text="hej och"),
            Segment(start=2.0, end=2.2, speaker="Björn", text="tack"),
        ],
        language="sv",
        model="external",
    )
    assert [(s.speaker, s.text) for s in result.segments] == [
        ("Anna", "hej och"),
        ("Björn", "tack"),
    ]
    assert "Anna: hej och" in result.text


def test_unlabelled_caller_segments_keep_the_diarizer_labels(tmp_path: Path):
    result = label_speakers(
        SwappedDiarizer(),
        tmp_path / "missing.wav",
        words=[Word(**w) for w in WORDS],
        segments=[Segment(start=0.0, end=2.2, text="hej och tack")],
        language="sv",
        model="external",
    )
    assert [s.speaker for s in result.segments] == ["SPEAKER_01", "SPEAKER_00"]


SEGMENT_ONLY = [Segment(start=0.0, end=2.7, text="hej och tack")]


def fake_aligner(audio_path: Path, segments: list[Segment], language: str) -> list[Word]:
    return [Word(**w) for w in WORDS]


def test_segments_only_are_force_aligned_into_word_level_splits(tmp_path: Path):
    # one input segment spans both diarization turns; aligned words split it in two
    result = label_speakers(
        FakeDiarizer(),
        tmp_path / "a.wav",
        words=[],
        segments=SEGMENT_ONLY,
        language="sv",
        model="kb-whisper-large",
        aligner=fake_aligner,
    )
    assert result.alignment == "forced"
    assert [(s.speaker, s.text) for s in result.segments] == [
        ("SPEAKER_00", "hej och"),
        ("SPEAKER_01", "tack"),
    ]
    assert result.model == "kb-whisper-large"


def test_whole_file_segment_is_force_aligned(tmp_path: Path):
    # Eneo rejecting a bad provider timeline sends one segment covering the whole
    # file; the aligner recovers word timestamps and the merge splits per speaker
    class LongDiarizer(FakeDiarizer):
        def diarize(self, audio_path: Path, *, speakers=None) -> list[Turn]:
            return [Turn(0.0, 30.0, "SPEAKER_00"), Turn(30.0, 65.0, "SPEAKER_01")]

    def whole_file_aligner(audio_path: Path, segments: list[Segment], language: str) -> list[Word]:
        assert [(s.start, s.end) for s in segments] == [(0.0, 65.0)]
        return [
            Word(word="hej", start=1.0, end=1.4),
            Word(word="där", start=2.0, end=2.3),
            Word(word="tack", start=40.0, end=40.4),
            Word(word="hej", start=41.0, end=41.2),
        ]

    result = label_speakers(
        LongDiarizer(),
        tmp_path / "a.wav",
        words=[],
        segments=[Segment(start=0.0, end=65.0, text="hej där tack hej")],
        language="sv",
        model="kb-whisper-large",
        aligner=whole_file_aligner,
    )
    assert result.alignment == "forced"
    assert [(s.speaker, s.text) for s in result.segments] == [
        ("SPEAKER_00", "hej där"),
        ("SPEAKER_01", "tack hej"),
    ]


def test_alignment_failure_fails_the_job(tmp_path: Path):
    # doctrine: never degrade to a segment-level merge — a broken alignment
    # stack must surface as a loud failure, not a coarser transcript
    def broken_aligner(audio_path: Path, segments: list[Segment], language: str) -> list[Word]:
        raise RuntimeError("no tokenizer for this language")

    with pytest.raises(RuntimeError, match="no tokenizer"):
        label_speakers(
            FakeDiarizer(),
            tmp_path / "a.wav",
            words=[],
            segments=SEGMENT_ONLY,
            language="sv",
            model="external",
            aligner=broken_aligner,
        )


def test_caller_words_win_over_the_aligner(tmp_path: Path):
    def exploding_aligner(audio_path: Path, segments: list[Segment], language: str) -> list[Word]:
        raise AssertionError("aligner must not run when words are supplied")

    result = label_speakers(
        FakeDiarizer(),
        tmp_path / "a.wav",
        words=[Word(**w) for w in WORDS],
        segments=SEGMENT_ONLY,
        language="sv",
        model="external",
        aligner=exploding_aligner,
    )
    assert result.alignment == "provider_words"


def test_prefer_alignment_sets_caller_words_aside(tmp_path: Path):
    # VEMSA_DIARIZE_PREFER_ALIGN: plausible caller words are set aside and the
    # transcript is force-aligned anyway; the (plausible) windows anchor it
    received: list[list[Segment]] = []

    def recording_aligner(audio_path: Path, segments: list[Segment], language: str) -> list[Word]:
        received.append(segments)
        return [Word(**w) for w in WORDS]

    result = label_speakers(
        FakeDiarizer(),
        tmp_path / "a.wav",
        words=[Word(word="fel", start=0.0, end=0.1), Word(word="ordning", start=0.2, end=0.3)],
        segments=SEGMENT_ONLY,
        language="sv",
        model="external",
        aligner=recording_aligner,
        prefer_alignment=True,
    )
    assert result.alignment == "forced"
    assert received == [SEGMENT_ONLY]
    assert "hej och" in result.text


def test_prefer_alignment_failure_fails_the_job(tmp_path: Path):
    # set-aside caller words are never resurrected: a failed alignment fails
    # the job rather than shipping timestamps the config said not to trust
    def broken_aligner(audio_path: Path, segments: list[Segment], language: str) -> list[Word]:
        raise RuntimeError("no tokenizer for this language")

    with pytest.raises(RuntimeError, match="no tokenizer"):
        label_speakers(
            FakeDiarizer(),
            tmp_path / "a.wav",
            words=[Word(**w) for w in WORDS],
            segments=SEGMENT_ONLY,
            language="sv",
            model="external",
            aligner=broken_aligner,
            prefer_alignment=True,
        )


def test_missing_aligner_fails_when_alignment_is_needed(tmp_path: Path):
    # segment-only input cannot be labelled word-precisely without alignment;
    # a missing align extra is a deployment error, not a reason to go coarse
    with pytest.raises(RuntimeError, match="align"):
        label_speakers(
            FakeDiarizer(),
            tmp_path / "a.wav",
            words=[],
            segments=SEGMENT_ONLY,
            language="sv",
            model="external",
            aligner=None,
        )


def compressed_words(count: int = 30, spacing: float = 0.1) -> list[Word]:
    """A decoder-heuristic timeline: ~10 words/s, humanly impossible."""
    return [
        Word(word=f"w{i}", start=i * spacing, end=i * spacing + spacing * 0.8) for i in range(count)
    ]


def test_words_plausible():
    # a normal ~2 words/s timeline passes
    normal = [Word(word=f"w{i}", start=i * 0.5, end=i * 0.5 + 0.3) for i in range(20)]
    assert words_plausible(normal)
    # a compressed timeline is rejected
    assert not words_plausible(compressed_words())
    # a long pause is excluded from speaking time, not used to mask compression
    with_pause = normal[:10] + [
        Word(word=f"p{i}", start=60.0 + i * 0.5, end=60.0 + i * 0.5 + 0.3) for i in range(10)
    ]
    assert words_plausible(with_pause)
    masked = compressed_words(30) + [Word(word="slut", start=60.0, end=60.3)]
    assert not words_plausible(masked)
    # too few words to judge: accepted
    assert words_plausible([Word(word="hej", start=0.0, end=0.1)])


def test_implausible_caller_words_are_discarded_for_alignment(tmp_path: Path):
    # the segment windows come from the same rejected timeline, so the aligner
    # must receive one whole-audio window, not the suspect windows
    received: list[list[Segment]] = []

    def recording_aligner(audio_path: Path, segments: list[Segment], language: str) -> list[Word]:
        received.append(segments)
        return [Word(**w) for w in WORDS]

    result = label_speakers(
        FakeDiarizer(),
        tmp_path / "a.wav",
        words=compressed_words(),
        segments=SEGMENT_ONLY,
        language="sv",
        model="external",
        aligner=recording_aligner,
    )
    assert result.alignment == "forced"
    assert len(received[0]) == 1
    assert received[0][0].start == 0.0
    assert received[0][0].text == "hej och tack"


def test_implausible_segment_windows_align_as_one_window(tmp_path: Path):
    # segments-only input whose windows imply an impossible speaking rate: the
    # provider timeline is bad even without words, so the windows are untrusted
    received: list[list[Segment]] = []

    def recording_aligner(audio_path: Path, segments: list[Segment], language: str) -> list[Word]:
        received.append(segments)
        return [Word(**w) for w in WORDS]

    compressed_segments = [
        Segment(start=0.0, end=1.0, text="ett två tre fyra fem sex sju åtta nio tio")
    ]
    result = label_speakers(
        FakeDiarizer(),
        tmp_path / "a.wav",
        words=[],
        segments=compressed_segments,
        language="sv",
        model="external",
        aligner=recording_aligner,
    )
    assert result.alignment == "forced"
    assert len(received[0]) == 1
    assert received[0][0].text == "ett två tre fyra fem sex sju åtta nio tio"


def test_locally_compressed_segment_windows_align_as_one_window(tmp_path: Path):
    # a clumpy timeline: one window crams a sentence into a second while a long
    # slow window drags the global average under the threshold — the per-segment
    # check must still reject the windows as alignment anchors
    received: list[list[Segment]] = []

    def recording_aligner(audio_path: Path, segments: list[Segment], language: str) -> list[Word]:
        received.append(segments)
        return [Word(**w) for w in WORDS]

    clumpy = [
        Segment(start=0.0, end=1.0, text="ett två tre fyra fem sex sju åtta nio tio"),
        Segment(start=10.0, end=60.0, text="elva tolv tretton fjorton femton"),
    ]
    result = label_speakers(
        FakeDiarizer(),
        tmp_path / "a.wav",
        words=[],
        segments=clumpy,
        language="sv",
        model="external",
        aligner=recording_aligner,
    )
    assert result.alignment == "forced"
    assert [(s.start, s.end) for s in received[0]] == [(0.0, 60.0)]


def test_implausible_words_without_segments_are_realigned_from_their_text(tmp_path: Path):
    # words-only input with a garbage timeline: the text survives in order and
    # is force-aligned against the whole audio — never shipped with the broken
    # timestamps, never dropped
    received: list[list[Segment]] = []

    def recording_aligner(audio_path: Path, segments: list[Segment], language: str) -> list[Word]:
        received.append(segments)
        return [Word(**w) for w in WORDS]

    result = label_speakers(
        FakeDiarizer(),
        tmp_path / "a.wav",
        words=compressed_words(),
        segments=[],
        language="sv",
        model="external",
        aligner=recording_aligner,
    )
    assert result.alignment == "forced"
    assert len(received[0]) == 1
    assert received[0][0].text.split() == [f"w{i}" for i in range(30)]


def test_build_segment_aligner_is_unconditional(settings: Settings, monkeypatch):
    # forced alignment is mandatory: no config gate, and a missing easyaligner
    # only logs at build time (jobs needing alignment then fail loudly at run time)
    monkeypatch.setattr(align, "alignment_available", lambda: True)
    assert align.build_segment_aligner(settings) is not None
    monkeypatch.setattr(align, "alignment_available", lambda: False)
    assert align.build_segment_aligner(settings) is not None


def test_diarize_only_engine_aligns_and_passes_speaker_bounds(settings: Settings, tmp_path: Path):
    diarizer = FakeDiarizer()
    engine = DiarizeOnlyEngine(settings, diarizer=diarizer)
    engine._segment_aligner = fake_aligner
    bounds = SpeakerBounds(max_speakers=2)
    result = engine.label_speakers(
        tmp_path / "a.wav",
        words=[],
        segments=SEGMENT_ONLY,
        language="sv",
        model="external",
        speakers=bounds,
    )
    assert result.alignment == "forced"
    assert len(result.segments) == 2
    assert diarizer.speaker_bounds == [bounds]


def test_prefer_align_setting_reaches_the_merge(settings: Settings, tmp_path: Path):
    # on by default: caller words are set aside and the transcript is realigned
    assert settings.diarize_prefer_align is True
    engine = DiarizeOnlyEngine(settings, diarizer=FakeDiarizer())
    engine._segment_aligner = fake_aligner
    caller_words = [Word(word="fel", start=0.0, end=0.1), Word(word="ordning", start=0.2, end=0.3)]
    result = engine.label_speakers(
        tmp_path / "a.wav",
        words=caller_words,
        segments=SEGMENT_ONLY,
        language="sv",
        model="external",
    )
    assert result.alignment == "forced"

    # the opt-out restores caller-words-first for callers that measure honestly
    settings.diarize_prefer_align = False
    engine = DiarizeOnlyEngine(settings, diarizer=FakeDiarizer())
    engine._segment_aligner = fake_aligner
    result = engine.label_speakers(
        tmp_path / "a.wav",
        words=caller_words,
        segments=SEGMENT_ONLY,
        language="sv",
        model="external",
    )
    assert result.alignment == "provider_words"


async def test_speaker_bounds_reach_the_engine_from_both_tasks(settings: Settings):
    engine = FakeEngine()
    async with api_client(settings, engine) as (client, _):
        response = await submit(
            client, task="diarize", language="sv", words=json.dumps(WORDS), max_speakers="3"
        )
        assert response.status_code == 202
        await poll_until(client, response.json()["job_id"], "completed")
        response = await submit(client, language="sv", num_speakers="2")
        assert response.status_code == 202
        await poll_until(client, response.json()["job_id"], "completed")

    assert engine.calls[0]["speakers"] == SpeakerBounds(max_speakers=3)
    assert engine.calls[1]["speakers"] == SpeakerBounds(num_speakers=2)


async def test_speaker_bounds_validation(settings: Settings):
    async with api_client(settings) as (client, _):
        # num_speakers excludes min/max
        response = await submit(client, num_speakers="2", max_speakers="3")
        assert response.status_code == 422
        # inverted range
        response = await submit(client, min_speakers="4", max_speakers="2")
        assert response.status_code == 422
        # below 1
        response = await submit(client, num_speakers="0")
        assert response.status_code == 422
        # bounds without diarization make no sense
        response = await submit(client, diarize="false", max_speakers="3")
        assert response.status_code == 422


def test_decodable_audio_passes_pcm_containers_through(tmp_path: Path):
    soundfile = pytest.importorskip("soundfile")
    path = tmp_path / "audio.wav"
    soundfile.write(path, [0.0] * 1600, 16000)

    resolved, is_temp = _decodable_audio(path)

    assert resolved == path
    assert not is_temp


def test_decodable_audio_transcodes_compressed_formats(tmp_path: Path, monkeypatch):
    # pyannote 4 decodes via torchcodec, which fails on formats libsndfile reads
    # fine (VBR mp3); soundfile-readable must not short-circuit the transcode
    soundfile = pytest.importorskip("soundfile")
    path = tmp_path / "audio.mp3"
    try:
        soundfile.write(path, [0.0] * 1600, 16000)
    except Exception:
        pytest.skip("libsndfile without mp3 write support")

    commands = []

    def fake_ffmpeg(cmd, **kwargs):
        commands.append(cmd)
        Path(cmd[-1]).write_bytes(b"")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr("vemsa.pipeline.diarize.subprocess.run", fake_ffmpeg)

    resolved, is_temp = _decodable_audio(path)

    assert is_temp
    assert resolved.parent == path.parent
    assert resolved.name.startswith("audio.mp3.") and resolved.name.endswith(".diarize.wav")
    assert commands and commands[0][0] == "ffmpeg"


def test_decodable_audio_temp_names_are_unique(tmp_path, monkeypatch):
    import subprocess

    from vemsa.pipeline.diarize import _decodable_audio

    path = tmp_path / "audio.mp3"
    path.write_bytes(b"not a wav")

    def fake_ffmpeg(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"RIFF")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr("vemsa.pipeline.diarize.subprocess.run", fake_ffmpeg)

    first, _ = _decodable_audio(path)
    second, _ = _decodable_audio(path)
    assert first != second
    assert first.exists() and second.exists()
