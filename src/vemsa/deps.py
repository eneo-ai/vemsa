from dataclasses import dataclass, field

from vemsa.config import Settings
from vemsa.jobs.queue import JobQueue
from vemsa.jobs.store import JobStore
from vemsa.pipeline.base import TranscriptionEngine


@dataclass
class AppDeps:
    """Shared runtime dependencies; store/queue/engine are populated by the app lifespan."""

    settings: Settings
    engine: TranscriptionEngine | None = None
    store: JobStore | None = None
    queue: JobQueue | None = field(default=None)

    @property
    def ready_store(self) -> JobStore:
        assert self.store is not None, "app lifespan has not run"
        return self.store

    @property
    def ready_queue(self) -> JobQueue:
        assert self.queue is not None, "app lifespan has not run"
        return self.queue
