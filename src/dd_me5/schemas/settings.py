from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from dd_me5.schemas.enums import Device


class Settings(BaseSettings):
    host: str = Field("0.0.0.0", validation_alias="HOST", description="Host to bind to")
    port: int = Field(8080, validation_alias="PORT", description="Port to bind to")
    model_dir: Path = Field(..., description="Path to model directory")
    device: Device = Field(Device.cuda, description="Device to use @ cpu, cuda")
    warmup: bool = Field(False, description="Warmup mode")
    max_batch_size: Optional[int] = Field(32, description="Maximum batch size")
    batch_timeout: Optional[float] = Field(0.01, description="Batch timeout")
    reload: Optional[bool] = Field(True, description="Reload mode")
    log_file: Optional[str] = Field(None, description="File to write logs to if desired")
    log_level: Optional[str] = Field("INFO", validation_alias="LOG_LEVEL", description="Log level")
    model_config = SettingsConfigDict(env_file=".env", protected_namespaces=[], use_enum_values=True)


settings = Settings()
