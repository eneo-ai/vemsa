import logging
from urllib.parse import urlsplit

from tolka.config import Settings
from tolka.jobs.postgres_store import PostgresJobStore
from tolka.jobs.store import JobStore, SqliteJobStore

logger = logging.getLogger(__name__)


def _redacted_target(database_url: str) -> str:
    parts = urlsplit(database_url)
    host = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""
    return f"{host}{port}{parts.path}"


async def open_job_store(settings: Settings) -> JobStore:
    if settings.database_url:
        store: JobStore = PostgresJobStore(settings.database_url)
        backend, target = "postgres", _redacted_target(settings.database_url)
    else:
        store = SqliteJobStore(settings.db_path)
        backend, target = "sqlite", str(settings.db_path)
    logger.info(
        "job store opened",
        extra={"event": "store.opened", "backend": backend, "target": target},
    )
    await store.open()
    return store
