import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel, ValidationError
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartException

from tolka.api.health import ReadinessResponse, load_readiness
from tolka.deps import AppDeps
from tolka.jobs.models import Job, JobRequest, JobStage, JobStatus, TranscriptionResult, new_job
from tolka.observability import (
    JOB_CANCELLATIONS,
    JOBS_SUBMITTED,
    QUEUE_REJECTIONS,
)
from tolka.pipeline.fetch import AudioTooLargeError, save_upload

router = APIRouter()
logger = logging.getLogger(__name__)


class JobSubmittedResponse(BaseModel):
    job_id: str
    status: JobStatus


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    stage: JobStage
    queue_position: int | None = None
    created_at: datetime
    error: str | None = None


class JobCancellationResponse(BaseModel):
    job_id: str
    status: JobStatus
    stage: JobStage
    cancellation_requested: bool


def _deps(request: Request) -> AppDeps:
    return request.app.state.deps


def _validation_error(exc: ValidationError) -> HTTPException:
    # include_context=False: a model_validator's raised ValueError rides along in
    # ctx as the exception object itself, which the JSON response cannot serialize
    return HTTPException(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail=exc.errors(include_url=False, include_context=False),
    )


async def _ensure_queue_capacity(deps: AppDeps, client_id: str) -> None:
    total = await deps.ready_store.count_active()
    client_total = await deps.ready_store.count_active(client_id=client_id)
    if total >= deps.settings.max_queued_jobs:
        QUEUE_REJECTIONS.labels("global").inc()
        logger.warning(
            "job rejected because the global queue is full",
            extra={
                "event": "queue.rejected",
                "scope": "global",
                "client_id": client_id,
                "active_jobs": total,
                "limit": deps.settings.max_queued_jobs,
            },
        )
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, detail="job queue is full")
    if client_total >= deps.settings.max_queued_jobs_per_client:
        QUEUE_REJECTIONS.labels("client").inc()
        logger.warning(
            "job rejected because the client queue is full",
            extra={
                "event": "queue.rejected",
                "scope": "client",
                "client_id": client_id,
                "active_jobs": client_total,
                "limit": deps.settings.max_queued_jobs_per_client,
            },
        )
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            detail="client has reached its active job limit",
        )


_TRANSCRIPT_FIELDS = ("words", "segments")  # size-capped against max_transcript_bytes
_JSON_FIELDS = (*_TRANSCRIPT_FIELDS, "vocabulary")


def _part_too_large(deps: AppDeps) -> HTTPException:
    return HTTPException(
        status.HTTP_413_CONTENT_TOO_LARGE,
        detail=f"a form part exceeds {deps.settings.max_transcript_bytes} bytes",
    )


def _admit_request(job_request: JobRequest, deps: AppDeps) -> None:
    if job_request.task == "transcribe" and deps.settings.resolve_engine() == "diarize":
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="this deployment only accepts task=diarize jobs",
        )
    if job_request.transcript_bytes() > deps.settings.max_transcript_bytes:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"transcript exceeds {deps.settings.max_transcript_bytes} bytes",
        )


async def _job_from_multipart(request: Request, deps: AppDeps, client_id: str) -> Job:
    # starlette caps text form parts at 1 MiB by default, which a transcript part
    # can legitimately exceed; raise the cap so our own 413 below is what binds
    # (file parts spool to disk and are bounded separately by max_audio_bytes).
    # A part blowing even the raised cap surfaces as starlette's own 400 (it
    # converts MultiPartException itself when an app is present) — normalize to
    # the contract's 413.
    try:
        form = await request.form(max_part_size=deps.settings.max_transcript_bytes + 65536)
    except MultiPartException as exc:
        raise _part_too_large(deps) from exc
    except StarletteHTTPException as exc:
        if exc.status_code == 400 and "exceeded maximum size" in str(exc.detail).lower():
            raise _part_too_large(deps) from exc
        raise
    upload = form.get("file")
    if not isinstance(upload, UploadFile):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, detail="multipart requests need a 'file' part"
        )
    fields: dict[str, object] = {
        key: value for key, value in form.items() if key != "file" and value != ""
    }
    # Multipart values are strings; these parts carry JSON arrays.
    for name in _JSON_FIELDS:
        raw = fields.get(name)
        if not isinstance(raw, str):
            continue
        if name in _TRANSCRIPT_FIELDS and len(raw.encode()) > deps.settings.max_transcript_bytes:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"transcript exceeds {deps.settings.max_transcript_bytes} bytes",
            )
        try:
            fields[name] = json.loads(raw)
        except ValueError as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, detail=f"'{name}' is not valid JSON"
            ) from exc
    try:
        job_request = JobRequest.model_validate(fields)
    except ValidationError as exc:
        raise _validation_error(exc) from exc
    _admit_request(job_request, deps)
    try:
        audio_path = await save_upload(
            upload,
            dest_dir=deps.settings.work_dir,
            max_bytes=deps.settings.max_audio_bytes,
        )
    except AudioTooLargeError as exc:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, detail=str(exc)) from exc
    return new_job(job_request, audio_path=str(audio_path), client_id=client_id)


async def _job_from_json(request: Request, deps: AppDeps, client_id: str) -> Job:
    try:
        job_request = JobRequest.model_validate(await request.json())
    except ValidationError as exc:
        raise _validation_error(exc) from exc
    except ValueError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, detail="request body is not valid JSON"
        ) from exc
    _admit_request(job_request, deps)
    if job_request.source_url is None:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="source_url is required (or upload a file via multipart)",
        )
    return new_job(job_request, client_id=client_id)


@router.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
async def create_job(request: Request) -> JobSubmittedResponse:
    deps = _deps(request)
    client_id: str = request.state.client_id
    await _ensure_queue_capacity(deps, client_id)
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/"):
        job = await _job_from_multipart(request, deps, client_id)
    else:
        job = await _job_from_json(request, deps, client_id)
    await deps.ready_store.create(job)
    JOBS_SUBMITTED.labels("upload" if job.audio_path else "url").inc()
    logger.info(
        "job submitted",
        extra={
            "event": "job.submitted",
            "job_id": job.id,
            "client_id": client_id,
            "task": job.request.task,
            "source_type": "upload" if job.audio_path else "url",
        },
    )
    if deps.queue is not None:
        deps.queue.notify()
    return JobSubmittedResponse(job_id=job.id, status=job.status)


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request) -> JobStatusResponse:
    deps = _deps(request)
    client_id: str = request.state.client_id
    job = await deps.ready_store.get(job_id, client_id=client_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="unknown job")
    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        stage=job.stage,
        queue_position=await deps.ready_store.queue_position(job.id, client_id=client_id),
        created_at=job.created_at,
        error=job.error,
    )


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str, request: Request, response: Response) -> JobCancellationResponse:
    deps = _deps(request)
    client_id: str = request.state.client_id
    existing = await deps.ready_store.get(job_id, client_id=client_id)
    if existing is None:
        JOB_CANCELLATIONS.labels("unknown").inc()
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="unknown job")

    if existing.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
        cancelled = await deps.ready_store.cancel(
            job_id,
            client_id=client_id,
            webhook_url=(
                str(existing.request.webhook_url) if existing.request.webhook_url else None
            ),
        )
        if cancelled is not None:
            response.status_code = status.HTTP_202_ACCEPTED
            JOB_CANCELLATIONS.labels(existing.status.value).inc()
            if cancelled.attempt == 0 and cancelled.audio_path:
                Path(cancelled.audio_path).unlink(missing_ok=True)
            logger.info(
                "job cancellation accepted",
                extra={
                    "event": "job.cancellation_accepted",
                    "job_id": job_id,
                    "client_id": client_id,
                    "previous_status": existing.status.value,
                    "stage": cancelled.stage.value,
                },
            )
            return JobCancellationResponse(
                job_id=cancelled.id,
                status=cancelled.status,
                stage=cancelled.stage,
                cancellation_requested=True,
            )
        existing = await deps.ready_store.get(job_id, client_id=client_id)
        if existing is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="unknown job")

    JOB_CANCELLATIONS.labels("already_terminal").inc()
    return JobCancellationResponse(
        job_id=existing.id,
        status=existing.status,
        stage=existing.stage,
        cancellation_requested=existing.status == JobStatus.CANCELLED,
    )


@router.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str, request: Request) -> TranscriptionResult:
    deps = _deps(request)
    client_id: str = request.state.client_id
    job = await deps.ready_store.get(job_id, client_id=client_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="unknown job")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail={"job_id": job.id, "status": job.status.value, "error": job.error},
        )
    result = await deps.ready_store.get_result(job_id, client_id=client_id)
    assert result is not None
    return result


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness(request: Request, response: Response) -> ReadinessResponse:
    snapshot = await load_readiness(_deps(request), client_id=request.state.client_id)
    if snapshot.status != "ready":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return snapshot
