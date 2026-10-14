"""task=timestamp: word timestamps and reader lines for a caller transcript, no speakers."""

import json
from pathlib import Path

import pytest

from conftest import FakeEngine, wait_for_status
from test_diarize_task import AUTH, api_client, poll_until, submit
from test_queue import running_queue, upload_job
from test_realign import RecordingAligner
from vemsa.config import Settings
from vemsa.jobs.models import JobRequest, JobStage, JobStatus, Segment, Word
from vemsa.jobs.store import JobStore
from vemsa.pipeline.diarize_only import DiarizeOnlyEngine
from vemsa.pipeline.fake import CannedEngine
from vemsa.pipeline.hybrid import HybridEngine
from vemsa.pipeline.render import render_text
from vemsa.pipeline.timestamp import timestamp_transcript
from vemsa.pipeline.transcribe import EasyTranscriberEngine
from vemsa.pipeline.whisper_api import OpenAIWhisperEngine

# eneo's chunk windows: measured by the caller, text only, no speakers
CHUNKS = [
    {"start": 0.0, "end": 4.0, "text": "Hej och välkomna. Vi börjar nu."},
    {"start": 4.0, "end": 8.0, "text": "Första punkten är budgeten."},
]
CHUNK_WORDS = " ".join(chunk["text"] for chunk in CHUNKS).split()


def chunk_segments() -> list[Segment]:
    return [Segment(**chunk) for chunk in CHUNKS]


class RefusingDiarizer:
    def diarize(self, audio_path: Path, *, speakers=None):
        raise AssertionError("task=timestamp must never diarize")

    def load(self) -> None:
        pass


# ---- request validation


def test_timestamp_request_defaults_to_no_speakers():
    request = JobRequest(task="timestamp", segments=CHUNKS)
    assert request.diarize is False and request.speaker_bounds() is None
    assert JobRequest(task="timestamp", segments=CHUNKS, diarize=False) == request
    # the worker reads the stored request back; it must validate unchanged
    assert JobRequest.model_validate_json(request.model_dump_json()) == request


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({}, "segments list with text"),
        ({"segments": []}, "segments list with text"),
        ({"segments": [{"start": 0.0, "end": 1.0, "text": "  "}]}, "segments list with text"),
        (
            {"segments": CHUNKS, "words": [{"word": "hej", "start": 0.0, "end": 0.4}]},
            "words are not accepted",
        ),
        (
            {"segments": [{**CHUNKS[0], "words": [{"word": "hej", "start": 0.0, "end": 0.4}]}]},
            "words are not accepted",
        ),
        ({"segments": [{**CHUNKS[0], "speaker": "Anna"}]}, "speakers"),
        ({"segments": [{**CHUNKS[0], "speaker_attribution": "assigned"}]}, "speaker-review"),
        ({"segments": [{**CHUNKS[0], "overlap_ids": ["overlap_0000"]}]}, "speaker-review"),
        ({"segments": CHUNKS, "speaker_review": {"version": 1}}, "speaker-review"),
        ({"segments": CHUNKS, "num_speakers": 2}, "speaker bounds"),
        ({"segments": CHUNKS, "min_speakers": 1, "max_speakers": 3}, "speaker bounds"),
        ({"segments": CHUNKS, "include_speaker_review": True}, "include_speaker_review"),
        ({"segments": CHUNKS, "vocabulary": ["Anna"]}, "vocabulary"),
        ({"segments": CHUNKS, "diarize": True}, "diarize=true"),
        ({"segments": [{"start": 2.0, "end": 1.0, "text": "hej"}]}, "timestamps"),
    ],
)
def test_timestamp_request_rejections(fields: dict, message: str):
    with pytest.raises(ValueError, match=message):
        JobRequest.model_validate({"task": "timestamp", **fields})


# ---- timestamp_transcript with a stubbed aligner


def test_words_are_timed_into_speakerless_lines(tmp_path: Path):
    aligner = RecordingAligner()
    result = timestamp_transcript(
        aligner, tmp_path / "missing.wav", segments=chunk_segments(), language="sv"
    )
    # plausible caller windows anchor the alignment as they are
    assert [(w.start, w.end, w.text) for w in aligner.windows[0]] == [
        (c["start"], c["end"], c["text"]) for c in CHUNKS
    ]
    assert result.alignment == "forced"
    assert result.speaker_review is None
    assert result.model == "external" and result.language == "sv"
    assert [w.word for s in result.segments for w in s.words] == CHUNK_WORDS
    assert all(s.speaker is None for s in result.segments)
    for line in result.segments:
        assert (line.start, line.end) == (line.words[0].start, line.words[-1].end)
        assert line.text == " ".join(w.word for w in line.words)
    assert result.text == render_text(result.segments)
    # audio unreadable: duration falls back to the last caller window's end
    assert result.duration_seconds == 8.0


def test_language_auto_is_reported_unknown(tmp_path: Path):
    result = timestamp_transcript(
        RecordingAligner(), tmp_path / "a.wav", segments=chunk_segments(), language="auto"
    )
    assert result.language == "unknown"


def test_implausible_windows_align_the_whole_audio_in_one_window(tmp_path: Path):
    # a window cramming ten words into a second cannot anchor the alignment:
    # the whole text goes against the whole audio instead
    aligner = RecordingAligner()
    clumpy = [
        Segment(start=0.0, end=1.0, text="ett två tre fyra fem sex sju åtta nio tio"),
        Segment(start=10.0, end=60.0, text="elva tolv tretton fjorton femton"),
    ]
    timestamp_transcript(aligner, tmp_path / "missing.wav", segments=clumpy, language="sv")
    assert [(w.start, w.end, w.text) for w in aligner.windows[0]] == [
        (
            0.0,
            60.0,
            "ett två tre fyra fem sex sju åtta nio tio elva tolv tretton fjorton femton",
        )
    ]


def test_five_minute_window_comes_back_as_bounded_lines(tmp_path: Path):
    # one chunk window of continuous speech: reader lines, never one paragraph
    text = " ".join(["det här är en mening som pågår en stund."] * 60)
    result = timestamp_transcript(
        RecordingAligner(),
        tmp_path / "missing.wav",
        segments=[Segment(start=0.0, end=300.0, text=text)],
        language="sv",
    )
    assert len(result.segments) > 10
    assert all(s.end - s.start <= 30.0 and len(s.words) <= 60 for s in result.segments)
    assert all(s.text.endswith(".") for s in result.segments)


def test_alignment_failure_propagates(tmp_path: Path):
    def broken_aligner(audio_path: Path, segments: list[Segment], language: str) -> list[Word]:
        raise RuntimeError("no tokenizer for this language")

    with pytest.raises(RuntimeError, match="no tokenizer"):
        timestamp_transcript(
            broken_aligner, tmp_path / "a.wav", segments=chunk_segments(), language="sv"
        )


def test_empty_alignment_fails(tmp_path: Path):
    def silent_aligner(audio_path: Path, segments: list[Segment], language: str) -> list[Word]:
        return []

    with pytest.raises(RuntimeError, match="no words"):
        timestamp_transcript(
            silent_aligner, tmp_path / "a.wav", segments=chunk_segments(), language="sv"
        )


# ---- every engine exposes the task


def engine_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        database_url="postgresql://unused/unused",
        whisper_api_base="http://whisper.local/v1",
        work_dir=tmp_path / "work",
        model_cache_dir=tmp_path / "models",
        attr_line_max_words=3,
    )


@pytest.mark.parametrize(
    "engine_class", [DiarizeOnlyEngine, HybridEngine, OpenAIWhisperEngine, EasyTranscriberEngine]
)
def test_every_engine_times_without_diarizing(tmp_path: Path, engine_class):
    engine = engine_class(engine_settings(tmp_path))
    engine._diarizer = RefusingDiarizer()
    engine._segment_aligner = RecordingAligner()
    stages: list[JobStage] = []
    result = engine.timestamp_transcript(
        tmp_path / "a.wav",
        segments=chunk_segments(),
        language="sv",
        model="external",
        on_stage=stages.append,
    )
    assert stages == [JobStage.ALIGNING]
    assert result.alignment == "forced" and result.speaker_review is None
    assert [w.word for s in result.segments for w in s.words] == CHUNK_WORDS
    assert all(s.speaker is None for s in result.segments)
    # the deployment's line tuning (VEMSA_ATTR_LINE_MAX_WORDS=3) reaches the grouping
    assert max(len(s.words) for s in result.segments) == 3


def test_canned_engine_times_speakerless_lines(tmp_path: Path):
    stages: list[JobStage] = []
    result = CannedEngine().timestamp_transcript(
        tmp_path / "a.wav",
        segments=chunk_segments(),
        language="auto",
        model="external",
        on_stage=stages.append,
    )
    assert stages == [JobStage.ALIGNING]
    assert result.alignment == "forced" and result.speaker_review is None
    assert result.language == "unknown" and result.model == "external"
    assert [w.word for s in result.segments for w in s.words] == CHUNK_WORDS
    assert all(s.speaker is None for s in result.segments)
    assert result.text == render_text(result.segments)


# ---- job lifecycle (PostgreSQL-backed)


async def test_multipart_timestamp_lifecycle(settings: Settings):
    engine = FakeEngine()
    async with api_client(settings, engine) as (client, _):
        response = await submit(
            client, task="timestamp", language="sv", segments=json.dumps(CHUNKS)
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        await poll_until(client, job_id, "completed")
        result = (await client.get(f"/v1/jobs/{job_id}/result", headers=AUTH)).json()

    assert result["alignment"] == "forced" and result["model"] == "external"
    assert result["speaker_review"] is None
    assert all(s["speaker"] is None and s["words"] for s in result["segments"])
    call = engine.calls[-1]
    assert call["task"] == "timestamp" and call["language"] == "sv"
    assert call["model"] == "external"
    assert [s.text for s in call["segments"]] == [c["text"] for c in CHUNKS]


async def test_timestamp_submission_validation_failures(settings: Settings):
    segments = json.dumps(CHUNKS)
    async with api_client(settings) as (client, _):
        assert (await submit(client, task="timestamp")).status_code == 422
        for extra in (
            {"diarize": "true"},
            {"max_speakers": "2"},
            {"include_speaker_review": "true"},
            {"vocabulary": '["Anna"]'},
            {"words": '[{"word": "hej", "start": 0, "end": 1}]'},
        ):
            response = await submit(client, task="timestamp", segments=segments, **extra)
            assert response.status_code == 422, extra
        labelled = json.dumps([{**CHUNKS[0], "speaker": "Anna"}])
        assert (await submit(client, task="timestamp", segments=labelled)).status_code == 422
        # diarize=false says what the task does anyway
        response = await submit(client, task="timestamp", segments=segments, diarize="false")
        assert response.status_code == 202


async def test_diarize_tier_accepts_timestamp(settings: Settings):
    # the task needs only the aligner, which the diarize-only tier has
    settings.engine = "diarize"
    settings.run_worker = False
    async with api_client(settings) as (client, _):
        response = await client.post(
            "/v1/jobs",
            json={
                "task": "timestamp",
                "source_url": "https://example.org/m.mp3",
                "segments": CHUNKS,
            },
            headers=AUTH,
        )
        assert response.status_code == 202


def timestamp_job(tmp_path: Path):
    return upload_job(tmp_path, task="timestamp", language="sv", segments=CHUNKS)


def timing_engine(settings: Settings, aligner) -> DiarizeOnlyEngine:
    engine = DiarizeOnlyEngine(settings, diarizer=RefusingDiarizer())
    engine._segment_aligner = aligner
    return engine


async def test_timestamp_job_times_words_without_diarizing(
    store: JobStore, settings: Settings, tmp_path: Path
):
    job, audio = timestamp_job(tmp_path)
    await store.create(job)
    async with running_queue(store, timing_engine(settings, RecordingAligner()), settings) as queue:
        queue.notify()
        await wait_for_status(store, job.id, JobStatus.COMPLETED)

    assert not audio.exists()
    result = await store.get_result(job.id)
    assert result is not None
    assert result.alignment == "forced" and result.speaker_review is None
    assert result.model == "external"
    assert [w.word for s in result.segments for w in s.words] == CHUNK_WORDS
    assert all(s.speaker is None for s in result.segments)
    row = await store.pool.fetchrow(  # type: ignore[attr-defined]
        "SELECT task, alignment, stage_seconds FROM job_stats WHERE job_id = $1", job.id
    )
    assert row["task"] == "timestamp" and row["alignment"] == "forced"
    # the job reported aligning and never diarizing
    assert set(json.loads(row["stage_seconds"])) == {"queued", "aligning"}


async def test_timestamp_alignment_failure_fails_the_job(
    store: JobStore, settings: Settings, tmp_path: Path
):
    def broken_aligner(audio_path: Path, segments: list[Segment], language: str) -> list[Word]:
        raise RuntimeError("no tokenizer for this language")

    job, audio = timestamp_job(tmp_path)
    await store.create(job)
    async with running_queue(store, timing_engine(settings, broken_aligner), settings) as queue:
        queue.notify()
        failed = await wait_for_status(store, job.id, JobStatus.FAILED)

    assert failed.failure_kind == "internal"
    assert failed.error == "RuntimeError: processing failed"
    assert not audio.exists()


async def test_timestamp_job_obeys_the_interpolated_share_floor(
    store: JobStore, settings: Settings, tmp_path: Path
):
    settings.align_max_interpolated_share = 0.5

    def interpolating_aligner(
        audio_path: Path, segments: list[Segment], language: str
    ) -> list[Word]:
        words = RecordingAligner()(audio_path, segments, language)
        return [w.model_copy(update={"probability": 0.0}) for w in words]

    job, _ = timestamp_job(tmp_path)
    await store.create(job)
    async with running_queue(
        store, timing_engine(settings, interpolating_aligner), settings
    ) as queue:
        queue.notify()
        failed = await wait_for_status(store, job.id, JobStatus.FAILED)

    assert failed.error is not None and "VEMSA_ALIGN_MAX_INTERPOLATED_SHARE" in failed.error
