import logging
from urllib.parse import urlsplit

from tolka.config import Settings
from tolka.jobs.postgres_store import PostgresJobStore
from tolka.jobs.store import JobStore

logger = logging.getLogger(__name__)


def _redacted_target(database_url: str) -> str:
    parts = urlsplit(database_url)
    host = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""
    return f"{host}{port}{parts.path}"


async def open_job_store(settings: Settings, *, role: str = "primary") -> JobStore:
    store = PostgresJobStore(settings.database_url)
    logger.info(
        "job store opened",
        extra={
            "event": "store.opened",
            "backend": "postgres",
            "target": _redacted_target(settings.database_url),
            "role": role,
        },
    )
    await store.open()
    return store
