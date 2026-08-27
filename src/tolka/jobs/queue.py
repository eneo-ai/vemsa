import asyncio
import concurrent.futures
import contextlib
import hashlib
import hmac
import json
import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import httpx

from tolka.config import Settings
from tolka.jobs.models import EXTERNAL_MODEL, Job, JobStage, JobStatus, WebhookOutboxEvent
from tolka.jobs.store import JobStore
from tolka.jobs.store_factory import open_job_store
from tolka.observability import (
    JOB_DURATION,
    JOB_STAGE_DURATION,
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


class JobCancelledError(RuntimeError):
    pass


class JobLeaseLostError(RuntimeError):
    pass


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
        # Lease renewal and the worker heartbeat run on a dedicated thread with
        # their own event loop and store connection, so a main loop starved by
        # GIL-heavy pipeline stages cannot let the job lease lapse mid-work.
        self._control_thread: threading.Thread | None = None
        self._control_loop: asyncio.AbstractEventLoop | None = None
        self._control_store: JobStore | None = None
        self._worker_heartbeat: concurrent.futures.Future[None] | None = None

    def notify(self) -> None:
        self._wakeup.set()

    @property
    def healthy(self) -> bool:
        return (
            bool(self._tasks)
            and all(not task.done() for task in self._tasks)
            and (self._control_thread is None or self._control_thread.is_alive())
        )

    async def start(self) -> None:
        self._stopping = False
        self._tasks = [
            asyncio.create_task(self._worker_loop(), name="tolka-worker"),
            asyncio.create_task(self._purge_loop(), name="tolka-purge"),
            asyncio.create_task(self._webhook_loop(), name="tolka-webhook"),
        ]
        if self._settings.database_url:
            # PostgreSQL: renew leases and heartbeat from a dedicated thread with
            # its own connection, immune to main-loop starvation.
            await self._start_control_thread()
            assert self._control_loop is not None
            self._worker_heartbeat = asyncio.run_coroutine_threadsafe(
                self._worker_heartbeat_loop(), self._control_loop
            )
        else:
            # SQLite is written for exactly one connection per process; a second
            # one contends on the write lock, so renewal stays on the main loop.
            self._tasks.append(
                asyncio.create_task(self._worker_heartbeat_loop(), name="tolka-worker-heartbeat")
            )
        self.notify()
        self._webhook_wakeup.set()
        logger.info(
            "worker ready, polling for jobs",
            extra={"event": "worker.ready", "worker_id": self._worker_id},
        )

    async def _start_control_thread(self) -> None:
        ready = threading.Event()
        failures: list[BaseException] = []
        self._control_thread = threading.Thread(
            target=self._run_control_loop,
            args=(ready, failures),
            name="tolka-lease-control",
            daemon=True,
        )
        self._control_thread.start()
        await asyncio.to_thread(ready.wait)
        if failures:
            self._control_thread = None
            raise RuntimeError("lease control thread failed to start") from failures[0]

    def _run_control_loop(self, ready: threading.Event, failures: list[BaseException]) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            store = loop.run_until_complete(open_job_store(self._settings, role="lease-control"))
        except BaseException as exc:  # surfaced to start() via `failures`
            failures.append(exc)
            ready.set()
            loop.close()
            return
        self._control_loop = loop
        self._control_store = store
        ready.set()
        try:
            loop.run_forever()
        finally:
            loop.run_until_complete(store.close())
            loop.close()

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
        if self._worker_heartbeat is not None:
            self._worker_heartbeat.cancel()
            self._worker_heartbeat = None
        if self._control_loop is not None:
            asyncio.run_coroutine_threadsafe(self._drain_control_loop(), self._control_loop)
        if self._control_thread is not None:
            await asyncio.to_thread(self._control_thread.join, 10.0)
            if self._control_thread.is_alive():
                logger.warning("lease control thread did not stop within timeout")
        self._control_thread = None
        self._control_loop = None
        self._control_store = None

    async def _drain_control_loop(self) -> None:
        current = asyncio.current_task()
        tasks = [task for task in asyncio.all_tasks() if task is not current]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        asyncio.get_running_loop().stop()

    async def _worker_loop(self) -> None:
        while not self._stopping:
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(self._settings.queue_poll_interval_s):
                    await self._wakeup.wait()
            self._wakeup.clear()
            try:
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
            except asyncio.CancelledError:
                raise
            except Exception:
                # A transient store error must not kill the claim loop; back off
                # one poll interval and try again.
                logger.exception("worker loop iteration failed; retrying")
                await asyncio.sleep(self._settings.queue_poll_interval_s)

    async def _run_job(self, job: Job) -> None:
        token = job_id_var.set(job.id)
        try:
            await self._process_job(job)
        finally:
            job_id_var.reset(token)

    async def _process_job(self, job: Job) -> None:
        started = time.perf_counter()
        audio_path = Path(job.audio_path) if job.audio_path else None
        delete_audio = False
        renewal: concurrent.futures.Future[None] | asyncio.Task[None] | None = None
        engine = self._settings.resolve_engine()
        current_stage = job.stage
        stage_started = time.perf_counter()
        loop = asyncio.get_running_loop()
        logger.info(
            "job started",
            extra={
                "event": "job.started",
                "job_id": job.id,
                "client_id": job.client_id,
                "task": job.request.task,
                "engine": engine,
                "attempt": job.attempt,
                "source_type": "upload" if job.audio_path else "url",
            },
        )

        async def _persist_stage(stage: JobStage) -> None:
            active = await self._store.set_stage(job.id, stage, worker_id=self._worker_id)
            if active:
                return
            current = await self._store.get(job.id)
            if current is not None and current.status == JobStatus.CANCELLED:
                raise JobCancelledError("job cancellation was requested")
            raise JobLeaseLostError("worker no longer owns the job")

        def _report_stage(stage: JobStage) -> None:
            nonlocal current_stage, stage_started
            if stage == current_stage:
                return
            future = asyncio.run_coroutine_threadsafe(_persist_stage(stage), loop)
            try:
                future.result(timeout=5.0)
            except TimeoutError as exc:
                future.cancel()
                raise RuntimeError("job stage update timed out") from exc
            JOB_STAGE_DURATION.labels(current_stage.value, engine, job.request.task).observe(
                time.perf_counter() - stage_started
            )
            logger.info(
                "job stage changed",
                extra={
                    "event": "job.stage_changed",
                    "job_id": job.id,
                    "client_id": job.client_id,
                    "from_stage": current_stage.value,
                    "stage": stage.value,
                    "engine": engine,
                    "task": job.request.task,
                },
            )
            current_stage = stage
            stage_started = time.perf_counter()

        def _record_cancelled() -> None:
            nonlocal delete_audio
            delete_audio = True
            JOBS_FINISHED.labels("cancelled", engine, job.request.task).inc()
            JOB_DURATION.labels("cancelled", engine, job.request.task).observe(
                time.perf_counter() - started
            )
            JOB_STAGE_DURATION.labels(current_stage.value, engine, job.request.task).observe(
                time.perf_counter() - stage_started
            )
            logger.info(
                "job stopped after cancellation",
                extra={
                    "event": "job.cancelled",
                    "job_id": job.id,
                    "client_id": job.client_id,
                    "stage": current_stage.value,
                    "engine": engine,
                    "task": job.request.task,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
            self._webhook_wakeup.set()

        try:
            if audio_path is None:
                assert job.request.source_url is not None
                logger.info(
                    "fetching source audio",
                    extra={
                        "event": "job.fetching_source",
                        "job_id": job.id,
                        "client_id": job.client_id,
                    },
                )
                audio_path = await fetch_url(
                    str(job.request.source_url),
                    dest_dir=self._settings.work_dir,
                    max_bytes=self._settings.max_audio_bytes,
                    timeout_s=self._settings.fetch_timeout_s,
                    allow_private=self._settings.allow_private_urls,
                    allowed_hosts=tuple(self._settings.source_allowed_hosts),
                )
                await self._store.set_audio_path(job.id, str(audio_path))
            renewal_coro = self._heartbeat_lease(
                job,
                store=self._control_store or self._store,
                engine=engine,
                started=started,
                current_stage=lambda: current_stage,
            )
            if self._control_loop is not None:
                renewal = asyncio.run_coroutine_threadsafe(renewal_coro, self._control_loop)
            else:
                renewal = asyncio.create_task(renewal_coro, name=f"tolka-lease-{job.id}")
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
                    on_stage=_report_stage,
                )
            else:
                result = await asyncio.to_thread(
                    self._engine.transcribe,
                    audio_path,
                    language=job.request.language,
                    model=job.request.model or self._settings.default_model,
                    diarize=job.request.diarize,
                    speakers=job.request.speaker_bounds(),
                    on_stage=_report_stage,
                )
            await _persist_stage(JobStage.FINALIZING)
            JOB_STAGE_DURATION.labels(current_stage.value, engine, job.request.task).observe(
                time.perf_counter() - stage_started
            )
            logger.info(
                "job stage changed",
                extra={
                    "event": "job.stage_changed",
                    "job_id": job.id,
                    "client_id": job.client_id,
                    "from_stage": current_stage.value,
                    "stage": JobStage.FINALIZING.value,
                    "engine": engine,
                    "task": job.request.task,
                },
            )
            current_stage = JobStage.FINALIZING
            stage_started = time.perf_counter()
            committed = await self._store.finish(
                job.id,
                result,
                worker_id=self._worker_id,
                webhook_url=str(job.request.webhook_url) if job.request.webhook_url else None,
            )
            if not committed:
                current = await self._store.get(job.id)
                if current is not None and current.status == JobStatus.CANCELLED:
                    _record_cancelled()
                    return
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
            JOB_STAGE_DURATION.labels(current_stage.value, engine, job.request.task).observe(
                time.perf_counter() - stage_started
            )
            self._webhook_wakeup.set()
        except JobCancelledError:
            _record_cancelled()
        except JobLeaseLostError:
            logger.warning("job %s work discarded after worker lease was lost", job.id)
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
                current = await self._store.get(job.id)
                if current is not None and current.status == JobStatus.CANCELLED:
                    _record_cancelled()
                    return
                logger.warning("job %s failure discarded after worker lease was lost", job.id)
                return
            delete_audio = True
            JOBS_FINISHED.labels("failed", self._settings.resolve_engine(), job.request.task).inc()
            JOB_DURATION.labels(
                "failed", self._settings.resolve_engine(), job.request.task
            ).observe(time.perf_counter() - started)
            self._webhook_wakeup.set()
        finally:
            if renewal is not None:
                renewal.cancel()
                if isinstance(renewal, asyncio.Task):
                    with contextlib.suppress(asyncio.CancelledError):
                        await renewal
            if delete_audio and audio_path is not None:
                audio_path.unlink(missing_ok=True)

    async def _heartbeat_lease(
        self,
        job: Job,
        *,
        store: JobStore,
        engine: str,
        started: float,
        current_stage: Callable[[], JobStage],
    ) -> None:
        while True:
            await asyncio.sleep(self._settings.lease_heartbeat_s)
            renewed = await store.renew_lease(job.id, self._worker_id, self._settings.job_lease_s)
            if not renewed:
                logger.warning("job %s worker lease could not be renewed", job.id)
                return
            logger.info(
                "job in progress",
                extra={
                    "event": "job.progress",
                    "job_id": job.id,
                    "client_id": job.client_id,
                    "task": job.request.task,
                    "engine": engine,
                    "stage": current_stage().value,
                    "elapsed_s": round(time.perf_counter() - started, 1),
                },
            )

    async def _worker_heartbeat_loop(self) -> None:
        store = self._control_store or self._store
        while True:
            await store.record_worker_heartbeat(self._worker_id)
            queued = await store.count_queued()
            logger.info(
                "worker heartbeat",
                extra={
                    "event": "worker.heartbeat",
                    "worker_id": self._worker_id,
                    "queued_jobs": queued,
                    "running_jobs": await store.count_active() - queued,
                },
            )
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
