import json
import logging
import time
import warnings
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import Request
from prometheus_client import Counter, Gauge, Histogram

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
# set by the worker around job processing so every pipeline log line carries the
# job id, including logs from code that never sees the Job (provider client, diarizer)
job_id_var: ContextVar[str | None] = ContextVar("job_id", default=None)

# third-party loggers that flood INFO/DEBUG with framework internals
_NOISY_LOGGERS = (
    "matplotlib",
    "fontTools",
    "speechbrain",
    "pytorch_lightning",
    "lightning.pytorch",
    "urllib3",
    "filelock",
)

HTTP_REQUESTS = Counter("tolka_http_requests_total", "HTTP requests", ("method", "route", "status"))
HTTP_DURATION = Histogram(
    "tolka_http_request_duration_seconds", "HTTP request duration", ("method", "route")
)
JOBS_SUBMITTED = Counter("tolka_jobs_submitted_total", "Submitted jobs", ("source_type",))
JOBS_FINISHED = Counter("tolka_jobs_finished_total", "Terminal jobs", ("status", "engine", "task"))
JOB_DURATION = Histogram(
    "tolka_job_processing_seconds",
    "End-to-end worker processing time",
    ("status", "engine", "task"),
)
JOB_STAGE_DURATION = Histogram(
    "tolka_job_stage_seconds",
    "Time spent in each persisted job stage",
    ("stage", "engine", "task"),
)
JOB_CANCELLATIONS = Counter(
    "tolka_job_cancellations_total",
    "Cancellation requests by outcome",
    ("outcome",),
)
JOB_ALIGNMENT = Counter(
    "tolka_job_alignment_total",
    "Completed jobs by word-timestamp rung — anything but 'forced' is a quality degradation",
    ("alignment", "engine", "task"),
)
QUEUE_REJECTIONS = Counter(
    "tolka_queue_rejections_total",
    "Jobs rejected at admission because a queue limit was reached",
    ("scope",),
)
QUEUE_DEPTH = Gauge("tolka_queue_depth", "Currently queued jobs")
WEBHOOK_DELIVERIES = Counter(
    "tolka_webhook_deliveries_total", "Webhook delivery attempts", ("outcome",)
)

_STANDARD_LOG_RECORD_FIELDS = set(logging.makeLogRecord({}).__dict__)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if request_id := request_id_var.get():
            payload["request_id"] = request_id
        if job_id := job_id_var.get():
            payload.setdefault("job_id", job_id)
        for key, value in record.__dict__.items():
            if key not in _STANDARD_LOG_RECORD_FIELDS and key not in {"message", "asctime"}:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str, log_format: str) -> None:
    handler = logging.StreamHandler()
    if log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.basicConfig(level=level, handlers=[handler], force=True)
    # emit warnings.warn() through the configured formatter instead of raw stderr
    logging.captureWarnings(True)
    # known upstream noise: pyannote and speechbrain both probe the deprecated
    # torchaudio backend API on import; nothing actionable on our side
    warnings.filterwarnings("ignore", message=r".*list_audio_backends has been deprecated.*")
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


async def request_observability_middleware(request: Request, call_next):
    supplied = request.headers.get("x-request-id", "")
    request_id = supplied if 0 < len(supplied) <= 128 else uuid4().hex
    token = request_id_var.set(request_id)
    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        route_object = request.scope.get("route")
        route = getattr(route_object, "path", "unmatched")
        elapsed = time.perf_counter() - started
        HTTP_REQUESTS.labels(request.method, route, str(status_code)).inc()
        HTTP_DURATION.labels(request.method, route).observe(elapsed)
        logging.getLogger("tolka.access").info(
            "request completed",
            extra={
                "event": "http.request",
                "method": request.method,
                "route": route,
                "status": status_code,
                "duration_ms": round(elapsed * 1000, 2),
                "client_id": getattr(request.state, "client_id", None),
            },
        )
        request_id_var.reset(token)
