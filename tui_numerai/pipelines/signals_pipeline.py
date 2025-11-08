"""LightGBM pipeline for Numerai Signals."""

from ..core import PipelineConfig
from ..core.config import CompetitionType
from ..core.pipeline import register_pipeline
from .lightgbm_pipeline import LightGBMPipeline


@register_pipeline("lightgbm_signals")
class SignalsPipeline(LightGBMPipeline):
    """LightGBM pipeline for Numerai Signals competition."""

    @classmethod
    def get_default_config(cls) -> PipelineConfig:
        """Get default configuration for Signals pipeline."""
        config = super().get_default_config()
        config.name = "lightgbm_signals"
        config.competition = CompetitionType.SIGNALS
        config.description = "LightGBM baseline for Numerai Signals"
        
        # Signals-specific parameters (uses stock market data)
        config.model_params.update({
            "n_estimators": 1500,
            "learning_rate": 0.015,
            "max_depth": 6,
            "num_leaves": 64,
        })
        
        return config
