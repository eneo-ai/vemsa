import asyncio
import contextlib
import hashlib
import hmac
import json
import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import httpx

from tolka.config import Settings
from tolka.jobs.models import EXTERNAL_MODEL, Job, WebhookOutboxEvent
from tolka.jobs.store import JobStore
from tolka.observability import (
    JOB_DURATION,
    JOBS_FINISHED,
    QUEUE_DEPTH,
    WEBHOOK_DELIVERIES,
    job_id_var,
)
from tolka.pipeline.base import TranscriptionEngine
from tolka.pipeline.fetch import fetch_url
from tolka.security import ForbiddenUrlError, validate_outbound_url

logger = logging.getLogger(__name__)

ORPHAN_MAX_AGE_S = 24 * 3600


class JobQueue:
    """Single worker draining the job store; the DB is the queue, this is only the pump."""

    def __init__(self, store: JobStore, engine: TranscriptionEngine, settings: Settings) -> None:
        self._store = store
        self._engine = engine
        self._settings = settings
        self._wakeup = asyncio.Event()
        self._webhook_wakeup = asyncio.Event()
        self._tasks: list[asyncio.Task[None]] = []
        self._worker_id = uuid4().hex
        self._stopping = False

    def notify(self) -> None:
        self._wakeup.set()

    @property
    def healthy(self) -> bool:
        return bool(self._tasks) and all(not task.done() for task in self._tasks)

    async def start(self) -> None:
        self._stopping = False
        self._tasks = [
            asyncio.create_task(self._worker_loop(), name="tolka-worker"),
            asyncio.create_task(self._purge_loop(), name="tolka-purge"),
            asyncio.create_task(self._worker_heartbeat_loop(), name="tolka-worker-heartbeat"),
            asyncio.create_task(self._webhook_loop(), name="tolka-webhook"),
        ]
        self.notify()
        self._webhook_wakeup.set()

    async def stop(self) -> None:
        self._stopping = True
        self.notify()
        self._webhook_wakeup.set()
        background_tasks = [
            task for task in self._tasks if task.get_name() not in {"tolka-worker", "tolka-webhook"}
        ]
        for task in background_tasks:
            task.cancel()
        try:
            async with asyncio.timeout(self._settings.shutdown_grace_s):
                await asyncio.gather(*self._tasks, return_exceptions=True)
        except TimeoutError:
            logger.warning("worker did not stop within graceful shutdown timeout")
            for task in self._tasks:
                task.cancel()
            for task in self._tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._tasks = []

    async def _worker_loop(self) -> None:
        while not self._stopping:
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(self._settings.queue_poll_interval_s):
                    await self._wakeup.wait()
            self._wakeup.clear()
            while (
                not self._stopping
                and (
                    job := await self._store.claim_next_queued(
                        worker_id=self._worker_id, lease_for_s=self._settings.job_lease_s
                    )
                )
                is not None
            ):
                await self._run_job(job)
            QUEUE_DEPTH.set(await self._store.count_queued())

    async def _run_job(self, job: Job) -> None:
        token = job_id_var.set(job.id)
        try:
            await self._process_job(job)
        finally:
            job_id_var.reset(token)

    async def _process_job(self, job: Job) -> None:
        logger.info("job %s started", job.id)
        started = time.perf_counter()
        audio_path = Path(job.audio_path) if job.audio_path else None
        delete_audio = False
        heartbeat: asyncio.Task[None] | None = None
        try:
            if audio_path is None:
                assert job.request.source_url is not None
                audio_path = await fetch_url(
                    str(job.request.source_url),
                    dest_dir=self._settings.work_dir,
                    max_bytes=self._settings.max_audio_bytes,
                    timeout_s=self._settings.fetch_timeout_s,
                    allow_private=self._settings.allow_private_urls,
                    allowed_hosts=tuple(self._settings.source_allowed_hosts),
                )
                await self._store.set_audio_path(job.id, str(audio_path))
            heartbeat = asyncio.create_task(
                self._heartbeat_lease(job.id), name=f"tolka-lease-{job.id}"
            )
            if job.request.task == "diarize":
                result = await asyncio.to_thread(
                    self._engine.label_speakers,
                    audio_path,
                    words=job.request.words or [],
                    segments=job.request.segments or [],
                    language=job.request.language,
                    # The result must never claim Tolka's default model ran.
                    model=job.request.model or EXTERNAL_MODEL,
                    speakers=job.request.speaker_bounds(),
                )
            else:
                result = await asyncio.to_thread(
                    self._engine.transcribe,
                    audio_path,
                    language=job.request.language,
                    model=job.request.model or self._settings.default_model,
                    diarize=job.request.diarize,
                    speakers=job.request.speaker_bounds(),
                )
            committed = await self._store.finish(
                job.id,
                result,
                worker_id=self._worker_id,
                webhook_url=str(job.request.webhook_url) if job.request.webhook_url else None,
            )
            if not committed:
                logger.warning("job %s result discarded after worker lease was lost", job.id)
                return
            delete_audio = True
            JOBS_FINISHED.labels(
                "completed", self._settings.resolve_engine(), job.request.task
            ).inc()
            JOB_DURATION.labels(
                "completed", self._settings.resolve_engine(), job.request.task
            ).observe(time.perf_counter() - started)
            logger.info(
                "job completed",
                extra={
                    "event": "job.completed",
                    "job_id": job.id,
                    "client_id": job.client_id,
                    "attempt": job.attempt,
                    "engine": self._settings.resolve_engine(),
                    "task": job.request.task,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
            self._webhook_wakeup.set()
        except asyncio.CancelledError:
            logger.warning("job %s interrupted; its lease will expire for retry", job.id)
            raise
        except Exception as exc:
            logger.error(
                "job failed",
                extra={
                    "event": "job.failed",
                    "job_id": job.id,
                    "client_id": job.client_id,
                    "attempt": job.attempt,
                    "engine": self._settings.resolve_engine(),
                    "task": job.request.task,
                    "error_type": type(exc).__name__,
                },
            )
            public_error = f"{type(exc).__name__}: processing failed"
            committed = await self._store.fail(
                job.id,
                public_error,
                worker_id=self._worker_id,
                webhook_url=str(job.request.webhook_url) if job.request.webhook_url else None,
            )
            if not committed:
                logger.warning("job %s failure discarded after worker lease was lost", job.id)
                return
            delete_audio = True
            JOBS_FINISHED.labels("failed", self._settings.resolve_engine(), job.request.task).inc()
            JOB_DURATION.labels(
                "failed", self._settings.resolve_engine(), job.request.task
            ).observe(time.perf_counter() - started)
            self._webhook_wakeup.set()
        finally:
            if heartbeat is not None:
                heartbeat.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat
            if delete_audio and audio_path is not None:
                audio_path.unlink(missing_ok=True)

    async def _heartbeat_lease(self, job_id: str) -> None:
        while True:
            await asyncio.sleep(self._settings.lease_heartbeat_s)
            renewed = await self._store.renew_lease(
                job_id, self._worker_id, self._settings.job_lease_s
            )
            if not renewed:
                logger.warning("job %s worker lease could not be renewed", job_id)
                return

    async def _worker_heartbeat_loop(self) -> None:
        while True:
            await self._store.record_worker_heartbeat(self._worker_id)
            await asyncio.sleep(self._settings.lease_heartbeat_s)

    async def _webhook_loop(self) -> None:
        while True:
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(self._settings.webhook_poll_interval_s):
                    await self._webhook_wakeup.wait()
            self._webhook_wakeup.clear()
            while event := await self._store.claim_webhook(
                self._worker_id, self._settings.job_lease_s
            ):
                await self._deliver_webhook(event)
            if self._stopping:
                return

    async def _deliver_webhook(self, event: WebhookOutboxEvent) -> None:
        try:
            await validate_outbound_url(
                event.url,
                allow_private=self._settings.allow_private_urls,
                allowed_hosts=self._settings.webhook_allowed_hosts,
                require_https=not self._settings.allow_insecure_webhooks,
            )
        except ForbiddenUrlError as exc:
            await self._store.reschedule_webhook(event.id, self._worker_id, str(exc), 0, 1)
            WEBHOOK_DELIVERIES.labels("rejected").inc()
            logger.error("webhook destination for job %s was rejected: %s", event.job_id, exc)
            return
        body = json.dumps(event.payload, separators=(",", ":"), sort_keys=True).encode()
        headers = {"Content-Type": "application/json"}
        if secret := self._settings.webhook_signing_secret:
            timestamp = str(int(time.time()))
            signature = hmac.new(
                secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256
            ).hexdigest()
            headers["X-Tolka-Timestamp"] = timestamp
            headers["X-Tolka-Signature-256"] = f"sha256={signature}"
        try:
            async with httpx.AsyncClient(timeout=self._settings.webhook_timeout_s) as client:
                response = await client.post(event.url, content=body, headers=headers)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            error_summary = (
                f"HTTP {exc.response.status_code}"
                if isinstance(exc, httpx.HTTPStatusError)
                else type(exc).__name__
            )
            terminal = await self._store.reschedule_webhook(
                event.id,
                self._worker_id,
                error_summary,
                min(2 ** min(event.attempt, 12), 3600),
                self._settings.webhook_max_attempts,
            )
            WEBHOOK_DELIVERIES.labels("exhausted" if terminal else "retry").inc()
            logger.warning(
                "webhook delivery for job %s failed on attempt %d: %s",
                event.job_id,
                event.attempt + 1,
                error_summary,
            )
            return
        await self._store.mark_webhook_delivered(event.id, self._worker_id)
        WEBHOOK_DELIVERIES.labels("success").inc()

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
