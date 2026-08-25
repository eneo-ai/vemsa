"""Development entry point: ``uv run start``.

Serves the API with the in-process worker using devcontainer defaults: the compose
Postgres (hostname ``postgres``), the fake engine, and a bare ``dev`` token. The
defaults are plain field defaults, so environment variables and ``.env`` override
them as usual (e.g. ``TOLKA_ENGINE=local uv run start``, or ``TOLKA_DATABASE_URL``
when running outside the devcontainer network).
"""

from typing import Annotated, Literal

from pydantic import Field
from pydantic_settings import NoDecode

from tolka.config import Settings
from tolka.main import create_app


class DevSettings(Settings):
    api_tokens: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["dev"])
    engine: Literal["auto", "local", "hybrid", "remote", "diarize", "fake"] = "fake"
    database_url: str | None = "postgresql://tolka:tolka@postgres/tolka"
    # download/load ML models during startup (mirrors production compose) so the
    # first job never pays the cold cost and gating problems fail loudly at boot
    preload_models: bool = True


def create_dev_app():
    return create_app(DevSettings())


def main() -> None:
    import uvicorn

    uvicorn.run(
        "tolka.dev:create_dev_app",
        factory=True,
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
