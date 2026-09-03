import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

if TYPE_CHECKING:
    from vemsa.pipeline.diarize import AttributionTuning

GIB = 1024**3
_CLIENT_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VEMSA_", env_file=".env", extra="ignore")

    # NoDecode: parsed from a comma-separated env value by the validator below,
    # not pydantic-settings' default JSON decoding
    api_tokens: Annotated[list[str], NoDecode] = Field(default_factory=list)
    environment: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "text"] = "json"
    hf_token: str | None = Field(
        default=None, validation_alias=AliasChoices("VEMSA_HF_TOKEN", "HF_TOKEN")
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
    # fallback CTC model for forced alignment when the job's language has no
    # entry in emissions_models
    emissions_model: str = "KBLab/wav2vec2-large-voxrex-swedish"
    # per-language CTC models for forced alignment, comma-separated lang=model;
    # a language without an entry aligns with emissions_model under a warning —
    # an acoustic-model mismatch is a silent quality cliff
    emissions_models: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["sv=KBLab/wav2vec2-large-voxrex-swedish"]
    )
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    # prefer the pipeline's exclusive diarization (one speaker active at a time,
    # chosen to match what an ASR system would transcribe) for word attribution;
    # disable to label against the raw, possibly overlapping turns instead
    diarize_exclusive: bool = True
    # task=diarize with caller-supplied words: set the words aside and force-align
    # the transcript locally anyway — word timestamps derived from the audio beat
    # whatever a provider's decoder heuristics produced. Disable only for callers
    # that measure their word timestamps honestly. Whenever alignment is needed
    # it is mandatory: a missing align extra or a failed alignment fails the job
    # instead of degrading to a segment-level merge.
    diarize_prefer_align: bool = True
    # quality floor: fail a job whose word-timestamp rung degrades below this
    # instead of completing with a coarser result (unset = degrade, never fail)
    min_alignment: Literal["forced", "provider_words", "segment_split", "segment_only"] | None = (
        None
    )

    # word→speaker attribution tuning (see AttributionTuning in pipeline/diarize.py
    # for what each knob does); defaults mirror the tested behaviour so these only
    # need touching when tuning against real audio
    attr_min_coverage: float = 0.25
    attr_island_max_words: int = 2
    attr_island_max_duration_s: float = 1.0
    attr_island_max_span_s: float = 3.0
    attr_inherit_max_gap_s: float = 2.0
    attr_boundary_max_words: int = 3
    attr_boundary_max_gap_s: float = 3.0
    attr_gap_split_s: float = 1.0
    attr_hard_gap_split_s: float = 15.0
    attr_min_segment_words: int = 1
    attr_min_segment_duration_s: float = 0.6
    attr_align_merge_gap_s: float = 0.5
    attr_align_window_pad_s: float = 0.5
    attr_relabel_min_share: float = 0.5
    # forced alignment falls back to linearly interpolated timestamps (word
    # probability 0.0) for a window whose text cannot be aligned — too long for
    # its audio, or characters the CTC vocabulary lacks. Fail a forced-rung job
    # whose share of interpolated words exceeds this (1.0 = never fail, only log)
    align_max_interpolated_share: float = 1.0

    model_cache_dir: Path = Path("data/models")
    work_dir: Path = Path("data/work")
    database_url: str

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
    # jobs one worker process runs at once; each has its own lease and stage stream
    worker_concurrency: int = 1
    # of those, how many may be inside a GPU stage at once (<= worker_concurrency);
    # 1 keeps the GPU serialized so results cannot differ from serial runs
    gpu_concurrency: int = 1
    # claims a job may consume before a GPU/host out-of-memory failure is final
    oom_max_attempts: int = 3
    oom_retry_delay_s: float = 30.0

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

    def attribution_tuning(self) -> "AttributionTuning":
        """The VEMSA_ATTR_* knobs as the pipeline's tuning object (imported lazily:
        pipeline.diarize is the pure attribution module and config must stay
        importable without it in the dependency picture)."""
        from vemsa.pipeline.diarize import AttributionTuning

        return AttributionTuning(
            min_coverage=self.attr_min_coverage,
            island_max_words=self.attr_island_max_words,
            island_max_duration_s=self.attr_island_max_duration_s,
            island_max_span_s=self.attr_island_max_span_s,
            inherit_max_gap_s=self.attr_inherit_max_gap_s,
            boundary_max_words=self.attr_boundary_max_words,
            boundary_max_gap_s=self.attr_boundary_max_gap_s,
            gap_split_s=self.attr_gap_split_s,
            hard_gap_split_s=self.attr_hard_gap_split_s,
            min_segment_words=self.attr_min_segment_words,
            min_segment_duration_s=self.attr_min_segment_duration_s,
            align_merge_gap_s=self.attr_align_merge_gap_s,
            align_window_pad_s=self.attr_align_window_pad_s,
            relabel_min_share=self.attr_relabel_min_share,
        )

    @property
    def emissions_model_overrides(self) -> dict[str, str]:
        """Per-language CTC model map parsed from ``emissions_models``."""
        overrides: dict[str, str] = {}
        for value in self.emissions_models:
            language, _, model = value.partition("=")
            overrides[language.strip().lower()] = model.strip()
        return overrides

    def emissions_model_for(self, language: str) -> tuple[str, bool]:
        """CTC model for a job language, and whether it was an explicit match.

        Unmatched languages fall back to ``emissions_model``; the caller should
        surface that as the quality warning it is (except for auto/unknown,
        where no better choice exists)."""
        model = self.emissions_model_overrides.get(language.strip().lower())
        if model:
            return model, True
        return self.emissions_model, False

    @field_validator(
        "api_tokens",
        "source_allowed_hosts",
        "webhook_allowed_hosts",
        "whisper_extra_form",
        "emissions_models",
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
        for value in self.emissions_models:
            language, separator, model = value.partition("=")
            if not separator or not language.strip() or not model.strip():
                raise ValueError("emissions_models entries must use language=model")
        if self.environment == "production":
            if not self.api_tokens:
                raise ValueError("VEMSA_API_TOKENS is required in production")
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
        for name in ("attr_min_coverage", "attr_relabel_min_share", "align_max_interpolated_share"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in ("attr_align_merge_gap_s", "attr_align_window_pad_s"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        for name in (
            "attr_island_max_words",
            "attr_island_max_duration_s",
            "attr_island_max_span_s",
            "attr_inherit_max_gap_s",
            "attr_boundary_max_words",
            "attr_boundary_max_gap_s",
            "attr_gap_split_s",
            "attr_hard_gap_split_s",
            "attr_min_segment_words",
            "attr_min_segment_duration_s",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.max_queued_jobs <= 0 or self.max_queued_jobs_per_client <= 0:
            raise ValueError("queue limits must be greater than zero")
        if self.max_queued_jobs_per_client > self.max_queued_jobs:
            raise ValueError("max_queued_jobs_per_client cannot exceed max_queued_jobs")
        if self.webhook_max_attempts <= 0:
            raise ValueError("webhook_max_attempts must be greater than zero")
        if self.worker_concurrency < 1:
            raise ValueError("worker_concurrency must be at least 1")
        if not 1 <= self.gpu_concurrency <= self.worker_concurrency:
            raise ValueError("gpu_concurrency must be between 1 and worker_concurrency")
        if self.oom_max_attempts < 1:
            raise ValueError("oom_max_attempts must be at least 1")
        if self.oom_retry_delay_s < 0:
            raise ValueError("oom_retry_delay_s cannot be negative")
        return self
