import asyncio
import signal

from vemsa.config import Settings
from vemsa.jobs.queue import JobQueue
from vemsa.jobs.store_factory import open_job_store
from vemsa.observability import configure_logging
from vemsa.pipeline.factory import build_engine


async def run_worker() -> None:
    settings = Settings()
    configure_logging(settings.log_level, settings.log_format)
    store = await open_job_store(settings)
    engine = build_engine(settings)
    if settings.preload_models:
        await asyncio.to_thread(engine.warm_up)
    queue = JobQueue(store, engine, settings)
    await queue.start()
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signal_name in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signal_name, stop_event.set)
    try:
        await stop_event.wait()
    finally:
        await queue.stop()
        await store.close()


def start() -> None:
    asyncio.run(run_worker())
