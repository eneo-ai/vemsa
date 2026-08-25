import hashlib
import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

GIB = 1024**3
_CLIENT_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TOLKA_", env_file=".env", extra="ignore")

    # NoDecode: parsed from a comma-separated env value by the validator below,
    # not pydantic-settings' default JSON decoding
    api_tokens: Annotated[list[str], NoDecode] = Field(default_factory=list)
    environment: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "text"] = "json"
    hf_token: str | None = Field(
        default=None, validation_alias=AliasChoices("TOLKA_HF_TOKEN", "HF_TOKEN")
    )

    # auto: hybrid when a whisper endpoint is configured, local otherwise.
    # local: in-process easytranscriber (whisper + forced alignment on this machine).
    # hybrid: remote whisper text + local easyaligner forced alignment.
    # remote: remote whisper only, provider timestamps as-is.
    # diarize: no ASR at all; only task=diarize jobs (caller-supplied transcripts).
    engine: Literal["auto", "local", "hybrid", "remote", "diarize", "fake"] = "auto"

    whisper_api_base: str = ""
    whisper_api_key: str | None = None
    whisper_timeout_s: float = 3600.0
    # extra key=value form fields for the transcription request, comma-separated —
    # for provider-specific switches (e.g. Berget AI needs align=true for word timestamps)
    whisper_extra_form: Annotated[list[str], NoDecode] = Field(default_factory=list)
    # language to send when a job requests "auto", for providers that require an
    # explicit language (e.g. GDM); empty omits the field so the provider auto-detects
    whisper_auto_language: str = ""
    default_model: str = "KBLab/kb-whisper-large"
    emissions_model: str = "KBLab/wav2vec2-large-voxrex-swedish"
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    # task=diarize with segments but no words: force-align the text locally for
    # word-precise speaker changes (requires the `align` extra; falls back to
    # segment-level labelling when unavailable or when alignment fails)
    diarize_force_align: bool = True

    model_cache_dir: Path = Path("data/models")
    work_dir: Path = Path("data/work")
    db_path: Path = Path("data/tolka.sqlite3")
    database_url: str | None = None

    max_audio_bytes: int = 2 * GIB
    # task=diarize: cap on the serialized words/segments a caller may attach
    max_transcript_bytes: int = 8 * 1024 * 1024
    fetch_timeout_s: float = 600.0
    retention_hours: float = 72.0
    purge_interval_s: float = 900.0
    webhook_timeout_s: float = 30.0
    webhook_poll_interval_s: float = 1.0
    webhook_max_attempts: int = 8
    queue_poll_interval_s: float = 1.0
    job_lease_s: float = 120.0
    lease_heartbeat_s: float = 30.0
    shutdown_grace_s: float = 30.0
    run_worker: bool = True
    worker_stale_s: float = 90.0

    mcp_max_audio_bytes: int = GIB
    mcp_sync_timeout_s: float = 900.0
    mcp_poll_interval_s: float = 2.0

    preload_models: bool = False
    allow_private_urls: bool = False
    source_allowed_hosts: Annotated[list[str], NoDecode] = Field(default_factory=list)
    webhook_allowed_hosts: Annotated[list[str], NoDecode] = Field(default_factory=list)
    allow_insecure_webhooks: bool = False
    webhook_signing_secret: str | None = None
    max_queued_jobs: int = 100
    max_queued_jobs_per_client: int = 10

    @property
    def token_clients(self) -> dict[str, str]:
        """Map bearer tokens to stable client identities.

        Named credentials use ``client_id=token``. Bare tokens remain supported for
        development and are assigned a non-secret fingerprint as their client id.
        """
        credentials: dict[str, str] = {}
        for value in self.api_tokens:
            client_id, separator, token = value.partition("=")
            if separator:
                credentials[token] = client_id
            else:
                fingerprint = hashlib.sha256(value.encode()).hexdigest()[:12]
                credentials[value] = f"token-{fingerprint}"
        return credentials

    def resolve_engine(self) -> str:
        if self.engine != "auto":
            return self.engine
        return "hybrid" if self.whisper_api_base else "local"

    @field_validator(
        "api_tokens",
        "source_allowed_hosts",
        "webhook_allowed_hosts",
        "whisper_extra_form",
        mode="before",
    )
    @classmethod
    def _split_tokens(cls, value: object) -> object:
        if isinstance(value, str):
            return [token.strip() for token in value.split(",") if token.strip()]
        return value

    @model_validator(mode="after")
    def _validate_production_settings(self) -> "Settings":
        for value in self.api_tokens:
            client_id, separator, token = value.partition("=")
            if separator and (not _CLIENT_ID.fullmatch(client_id) or not token):
                raise ValueError(
                    "named API credentials must use client_id=token with a valid client id"
                )
        for value in self.whisper_extra_form:
            key, separator, _ = value.partition("=")
            if not separator or not key:
                raise ValueError("whisper_extra_form entries must use key=value")
        if self.environment == "production":
            if not self.api_tokens:
                raise ValueError("TOLKA_API_TOKENS is required in production")
            if any("=" not in value for value in self.api_tokens):
                raise ValueError("production API credentials must be named using client_id=token")
            if self.engine == "fake":
                raise ValueError("the fake transcription engine is not allowed in production")
            if self.engine == "diarize" and not self.hf_token:
                raise ValueError("the diarize tier requires HF_TOKEN for the gated pyannote models")
        if self.job_lease_s <= self.lease_heartbeat_s * 2:
            raise ValueError("job_lease_s must be more than twice lease_heartbeat_s")
        for name in (
            "queue_poll_interval_s",
            "webhook_poll_interval_s",
            "lease_heartbeat_s",
            "shutdown_grace_s",
            "worker_stale_s",
            "retention_hours",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.max_transcript_bytes <= 0:
            raise ValueError("max_transcript_bytes must be greater than zero")
        if self.max_queued_jobs <= 0 or self.max_queued_jobs_per_client <= 0:
            raise ValueError("queue limits must be greater than zero")
        if self.max_queued_jobs_per_client > self.max_queued_jobs:
            raise ValueError("max_queued_jobs_per_client cannot exceed max_queued_jobs")
        if self.webhook_max_attempts <= 0:
            raise ValueError("webhook_max_attempts must be greater than zero")
        return self
