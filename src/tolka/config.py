from pathlib import Path
from typing import Annotated

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

GIB = 1024**3


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TOLKA_", env_file=".env", extra="ignore")

    # NoDecode: parsed from a comma-separated env value by the validator below,
    # not pydantic-settings' default JSON decoding
    api_tokens: Annotated[list[str], NoDecode] = Field(default_factory=list)
    hf_token: str | None = Field(
        default=None, validation_alias=AliasChoices("TOLKA_HF_TOKEN", "HF_TOKEN")
    )

    default_model: str = "KBLab/kb-whisper-large"
    emissions_model: str = "KBLab/wav2vec2-large-voxrex-swedish"
    diarization_model: str = "pyannote/speaker-diarization-3.1"

    model_cache_dir: Path = Path("data/models")
    work_dir: Path = Path("data/work")
    db_path: Path = Path("data/tolka.sqlite3")

    max_audio_bytes: int = 2 * GIB
    fetch_timeout_s: float = 600.0
    retention_hours: float = 72.0
    purge_interval_s: float = 900.0
    webhook_timeout_s: float = 30.0

    mcp_max_audio_bytes: int = GIB
    mcp_sync_timeout_s: float = 900.0
    mcp_poll_interval_s: float = 2.0

    preload_models: bool = False
    allow_private_urls: bool = False
    fake_engine: bool = False

    @field_validator("api_tokens", mode="before")
    @classmethod
    def _split_tokens(cls, value: object) -> object:
        if isinstance(value, str):
            return [token.strip() for token in value.split(",") if token.strip()]
        return value
