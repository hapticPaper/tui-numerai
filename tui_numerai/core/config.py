"""Configuration models using Pydantic."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CompetitionType(str, Enum):
    """Supported competition types."""

    NUMERAI = "numerai"
    CRYPTO = "crypto"
    SIGNALS = "signals"


class CompetitionConfig(BaseModel):
    """Configuration for a specific competition."""

    competition: CompetitionType
    data_dir: Path = Field(default_factory=lambda: Path("./data"))
    models_dir: Path = Field(default_factory=lambda: Path("./models"))
    predictions_dir: Path = Field(default_factory=lambda: Path("./predictions"))

    @field_validator("data_dir", "models_dir", "predictions_dir", mode="before")
    @classmethod
    def ensure_path(cls, v: Any) -> Path:
        if isinstance(v, str):
            return Path(v)
        return v

    def ensure_directories(self) -> None:
        """Create all required directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)


class PipelineConfig(BaseModel):
    """Base configuration for a pipeline."""

    name: str
    competition: CompetitionType
    version: str = "1.0.0"
    description: str = ""

    # Training parameters
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None

    # Model parameters (flexible for different models)
    model_params: Dict[str, Any] = Field(default_factory=dict)

    # Feature configuration
    feature_set: Optional[str] = None
    neutralize_features: bool = False

    # Target configuration
    target_name: str = "target"

    # Output configuration
    save_predictions: bool = True
    save_model: bool = True

    # Logging configuration
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    use_influxdb: bool = False
    influxdb_url: Optional[str] = None

    model_config = ConfigDict(extra="allow")  # Allow additional fields for flexibility


class RunConfig(BaseModel):
    """Configuration for a specific training run."""

    run_id: str
    pipeline_name: str
    pipeline_version: str
    competition: CompetitionType
    pipeline_config: PipelineConfig

    # Run metadata
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Run paths
    run_dir: Path
    model_path: Optional[Path] = None
    predictions_path: Optional[Path] = None

    # Run state
    status: str = "created"  # created, running, completed, failed, resumed
    resume_from: Optional[str] = None  # run_id to resume from

    # Metrics storage
    metrics: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("run_dir", mode="before")
    @classmethod
    def ensure_path(cls, v: Any) -> Path:
        if isinstance(v, str):
            return Path(v)
        return v

    def ensure_run_directory(self) -> None:
        """Create the run directory structure."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        (self.run_dir / "logs").mkdir(exist_ok=True)
        (self.run_dir / "metrics").mkdir(exist_ok=True)
        (self.run_dir / "plots").mkdir(exist_ok=True)
