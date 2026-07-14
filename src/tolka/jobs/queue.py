import asyncio
import contextlib
import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from tolka.config import Settings
from tolka.jobs.models import Job, JobStatus, TranscriptionResult
from tolka.jobs.store import JobStore
from tolka.pipeline.base import TranscriptionEngine
from tolka.pipeline.fetch import fetch_url

logger = logging.getLogger(__name__)

ORPHAN_MAX_AGE_S = 24 * 3600


class JobQueue:
    """Single worker draining the job store; the DB is the queue, this is only the pump."""

    webhook_retry_delays: tuple[float, ...] = (1.0, 5.0)

    def __init__(self, store: JobStore, engine: TranscriptionEngine, settings: Settings) -> None:
        self._store = store
        self._engine = engine
        self._settings = settings
        self._wakeup = asyncio.Event()
        self._tasks: list[asyncio.Task[None]] = []

    def notify(self) -> None:
        self._wakeup.set()

    async def start(self) -> None:
        requeued = await self._store.requeue_stuck()
        if requeued:
            logger.info("requeued %d jobs stuck in running state", requeued)
        self._tasks = [
            asyncio.create_task(self._worker_loop(), name="tolka-worker"),
            asyncio.create_task(self._purge_loop(), name="tolka-purge"),
        ]
        self.notify()

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks = []

    async def _worker_loop(self) -> None:
        while True:
            await self._wakeup.wait()
            self._wakeup.clear()
            while (job := await self._store.claim_next_queued()) is not None:
                await self._run_job(job)

    async def _run_job(self, job: Job) -> None:
        logger.info("job %s started", job.id)
        audio_path = Path(job.audio_path) if job.audio_path else None
        try:
            if audio_path is None:
                assert job.request.source_url is not None
                audio_path = await fetch_url(
                    str(job.request.source_url),
                    dest_dir=self._settings.work_dir,
                    max_bytes=self._settings.max_audio_bytes,
                    timeout_s=self._settings.fetch_timeout_s,
                    allow_private=self._settings.allow_private_urls,
                )
                await self._store.set_audio_path(job.id, str(audio_path))
            result = await asyncio.to_thread(
                self._engine.transcribe,
                audio_path,
                language=job.request.language,
                model=job.request.model or self._settings.default_model,
                diarize=job.request.diarize,
            )
            await self._store.finish(job.id, result)
            logger.info("job %s completed", job.id)
            await self._notify_webhook(job, JobStatus.COMPLETED, result=result)
        except Exception as exc:
            logger.exception("job %s failed", job.id)
            await self._store.fail(job.id, f"{type(exc).__name__}: {exc}")
            await self._notify_webhook(job, JobStatus.FAILED, error=str(exc))
        finally:
            if audio_path is not None:
                audio_path.unlink(missing_ok=True)

    async def _notify_webhook(
        self,
        job: Job,
        status: JobStatus,
        *,
        result: TranscriptionResult | None = None,
        error: str | None = None,
    ) -> None:
        if job.request.webhook_url is None:
            return
        payload: dict[str, object] = {"job_id": job.id, "status": status.value}
        if result is not None:
            payload["result"] = result.model_dump(mode="json")
        if error is not None:
            payload["error"] = error
        url = str(job.request.webhook_url)
        delays = (0.0, *self.webhook_retry_delays)
        for attempt, delay in enumerate(delays, start=1):
            if delay:
                await asyncio.sleep(delay)
            try:
                async with httpx.AsyncClient(timeout=self._settings.webhook_timeout_s) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                return
            except httpx.HTTPError as exc:
                logger.warning(
                    "webhook delivery for job %s failed (attempt %d/%d): %s",
                    job.id,
                    attempt,
                    len(delays),
                    exc,
                )
        logger.error("webhook delivery for job %s gave up after %d attempts", job.id, len(delays))

    async def _purge_loop(self) -> None:
        while True:
            await asyncio.sleep(self._settings.purge_interval_s)
            try:
                await self.purge_once()
            except Exception:
                logger.exception("retention purge failed")

    async def purge_once(self) -> None:
        cutoff = datetime.now(UTC) - timedelta(hours=self._settings.retention_hours)
        purged = await self._store.purge_older_than(cutoff)
        for job in purged:
            if job.audio_path:
                Path(job.audio_path).unlink(missing_ok=True)
        if purged:
            logger.info("purged %d jobs past retention", len(purged))
        self._sweep_orphans()

    def _sweep_orphans(self) -> None:
        work_dir = self._settings.work_dir
        if not work_dir.is_dir():
            return
        threshold = time.time() - ORPHAN_MAX_AGE_S
        for path in work_dir.iterdir():
            try:
                if path.is_file() and path.stat().st_mtime < threshold:
                    path.unlink(missing_ok=True)
            except OSError:
                logger.warning("could not sweep orphaned file %s", path)
