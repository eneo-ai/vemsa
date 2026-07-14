from dataclasses import dataclass, field

from tolka.config import Settings
from tolka.jobs.queue import JobQueue
from tolka.jobs.store import SqliteJobStore
from tolka.pipeline.base import TranscriptionEngine


@dataclass
class AppDeps:
    """Shared runtime dependencies; store/queue/engine are populated by the app lifespan."""

    settings: Settings
    engine: TranscriptionEngine | None = None
    store: SqliteJobStore | None = None
    queue: JobQueue | None = field(default=None)

    @property
    def ready_store(self) -> SqliteJobStore:
        assert self.store is not None, "app lifespan has not run"
        return self.store

    @property
    def ready_queue(self) -> JobQueue:
        assert self.queue is not None, "app lifespan has not run"
        return self.queue
