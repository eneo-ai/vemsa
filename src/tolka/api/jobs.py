from datetime import datetime

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ValidationError
from starlette.datastructures import UploadFile

from tolka.deps import AppDeps
from tolka.jobs.models import Job, JobRequest, JobStatus, TranscriptionResult, new_job
from tolka.observability import JOBS_SUBMITTED
from tolka.pipeline.fetch import AudioTooLargeError, save_upload

router = APIRouter()


class JobSubmittedResponse(BaseModel):
    job_id: str
    status: JobStatus


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    error: str | None = None


def _deps(request: Request) -> AppDeps:
    return request.app.state.deps


def _validation_error(exc: ValidationError) -> HTTPException:
    return HTTPException(
        status.HTTP_422_UNPROCESSABLE_CONTENT, detail=exc.errors(include_url=False)
    )


async def _ensure_queue_capacity(deps: AppDeps, client_id: str) -> None:
    total = await deps.ready_store.count_active()
    client_total = await deps.ready_store.count_active(client_id=client_id)
    if total >= deps.settings.max_queued_jobs:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, detail="job queue is full")
    if client_total >= deps.settings.max_queued_jobs_per_client:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            detail="client has reached its active job limit",
        )


async def _job_from_multipart(request: Request, deps: AppDeps, client_id: str) -> Job:
    form = await request.form()
    upload = form.get("file")
    if not isinstance(upload, UploadFile):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, detail="multipart requests need a 'file' part"
        )
    fields = {key: value for key, value in form.items() if key != "file" and value != ""}
    try:
        job_request = JobRequest.model_validate(fields)
    except ValidationError as exc:
        raise _validation_error(exc) from exc
    try:
        audio_path = await save_upload(
            upload,
            dest_dir=deps.settings.work_dir,
            max_bytes=deps.settings.max_audio_bytes,
        )
    except AudioTooLargeError as exc:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, detail=str(exc)) from exc
    return new_job(job_request, audio_path=str(audio_path), client_id=client_id)


async def _job_from_json(request: Request, client_id: str) -> Job:
    try:
        job_request = JobRequest.model_validate(await request.json())
    except ValidationError as exc:
        raise _validation_error(exc) from exc
    except ValueError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, detail="request body is not valid JSON"
        ) from exc
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
        job = await _job_from_json(request, client_id)
    await deps.ready_store.create(job)
    JOBS_SUBMITTED.labels("upload" if job.audio_path else "url").inc()
    if deps.queue is not None:
        deps.queue.notify()
    return JobSubmittedResponse(job_id=job.id, status=job.status)


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request) -> JobStatusResponse:
    job = await _deps(request).ready_store.get(job_id, client_id=request.state.client_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="unknown job")
    return JobStatusResponse(
        job_id=job.id, status=job.status, created_at=job.created_at, error=job.error
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
