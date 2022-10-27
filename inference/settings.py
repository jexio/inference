import enum
from pathlib import Path
from tempfile import gettempdir
from typing import Optional

from pydantic import BaseSettings, Field

from inference.utils import get_value

TEMP_DIR = Path(gettempdir())


class LogLevel(str, enum.Enum):  # noqa: WPS600
    """Possible log levels."""

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


class TritonEngineSettings(BaseSettings):
    """DTO class Triton inference server settings.

    Attributes:
        url: The inference server URL, e.g. 'localhost:8001'.
        verbose: If True generate verbose output. Default value is False.
        ssl: If True use SSL encrypted secure channel. Default is False.
        root_certificates: File holding the PEM-encoded root certificates as a byte
            string, or None to retrieve them from a default location
            chosen by gRPC runtime. The option is ignored if `ssl`
            is False. Default is None.
        private_key: File holding the PEM-encoded private key as a byte string,
            or None if no private key should be used. The option is
            ignored if `ssl` is False. Default is None.
        certificate_chain: File holding PEM-encoded certificate chain as a byte string
            to use or None if no certificate chain should be used. The
            option is ignored if `ssl` is False. Default is None.
    """

    url: str
    verbose: bool = Field(default=False)
    ssl: bool = Field(default=False)
    root_certificates: Optional[str] = Field(default=None)
    private_key: Optional[str] = Field(default=None)
    certificate_chain: Optional[str] = Field(default=None)

    class Config:
        env_file = ".env"
        env_prefix = "INFERENCE_TRITON_ENGINE"
        env_file_encoding = "utf-8"

    @classmethod
    def from_environment_variables(cls) -> "TritonEngineSettings":
        """Create a new instance of TritonEngineSettings from environment variables.

        Returns:
            A new instance of TritonEngineSettings.
        """
        return cls(
            url=get_value("TRITON_URL"),
            verbose=get_value("TRITON_VERBOSE", bool, False),
            ssl=get_value("TRITON_SSL", bool, False),
            root_certificates=get_value("TRITON_CERTIFICATES", lambda x: x if x != "None" else None),
            private_key=get_value("TRITON_PRIVATE_KEY", lambda x: x if x != "None" else None),
            certificate_chain=get_value("TRITON_CERTIFICATE_CHAIN", lambda x: x if x != "None" else None),
        )


class TritonExecutorSettings(BaseSettings):
    """DTO class Triton inference client settings.

    Attributes:
        timeout: The maximum end-to-end time, in seconds, the request is allowed to take.
        use_shared_memory:  Whether to use CUDA Shared IPC Memory for transferring data to Triton.
    """

    timeout: Optional[float] = Field(default=None)
    use_shared_memory: bool = Field(default=False)

    class Config:
        env_file = ".env"
        env_prefix = "INFERENCE_TRITON_EXECUTOR"
        env_file_encoding = "utf-8"

    @classmethod
    def from_environment_variables(cls) -> "TritonExecutorSettings":
        """Create a new instance of TritonExecutorSettings from environment variables.

        Returns:
            A new instance of TritonExecutorSettings.
        """
        return cls(
            client_timeout=get_value("TRITON_CLIENT_TIMEOUT", float),
            use_shared_memory=get_value("TRITON_USE_SHARED_MEMORY", bool),
        )


class TritonRunnerPolicy(BaseSettings):
    """DTO class Triton inference client settings.

    Attributes:
        timeout: The maximum end-to-end time, in seconds, the request is allowed to take.
        use_shared_memory:  Whether to use CUDA Shared IPC Memory for transferring data to Triton.
    """

    tries: int = Field(default=2)
    delay: float = Field(default=0.5, ge=0)
    max_delay: Optional[float] = Field(default=None, ge=1)
    backoff: int = Field(default=1, ge=1)
    jitter_min: int = Field(default=0, ge=0)
    jitter_max: int = Field(default=0, ge=0)

    class Config:
        env_file = ".env"
        env_prefix = "INFERENCE_TRITON_POLICY"
        env_file_encoding = "utf-8"


class Settings(BaseSettings):
    """
    Application settings.

    These parameters can be configured
    with environment variables.
    """

    host: str = "127.0.0.1"
    port: int = 8000
    # quantity of workers for uvicorn
    workers_count: int = 1
    # Enable uvicorn reloading
    reload: bool = False

    # Current environment
    environment: str = "dev"

    log_level: LogLevel = LogLevel.INFO

    # This variable is used to define
    # multiproc_dir. It's required for [uvi|guni]corn projects.
    prometheus_dir: Path = TEMP_DIR / "prom"

    # Grpc endpoint for opentelemetry.
    # E.G. http://localhost:4317
    opentelemetry_endpoint: Optional[str] = None

    class Config:
        env_file = ".env"
        env_prefix = "INFERENCE_"
        env_file_encoding = "utf-8"


settings = Settings()
runner_policy = TritonRunnerPolicy()
executor_settings = TritonExecutorSettings()
#engine_settings = TritonEngineSettings()
