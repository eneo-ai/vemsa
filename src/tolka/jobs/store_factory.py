from tolka.config import Settings
from tolka.jobs.postgres_store import PostgresJobStore
from tolka.jobs.store import JobStore, SqliteJobStore


async def open_job_store(settings: Settings) -> JobStore:
    if settings.database_url:
        store: JobStore = PostgresJobStore(settings.database_url)
    else:
        store = SqliteJobStore(settings.db_path)
    await store.open()
    return store
