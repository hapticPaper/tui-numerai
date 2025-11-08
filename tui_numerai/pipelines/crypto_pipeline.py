"""LightGBM pipeline for Numerai Crypto."""

from ..core import PipelineConfig
from ..core.config import CompetitionType
from ..core.pipeline import register_pipeline
from .lightgbm_pipeline import LightGBMPipeline


@register_pipeline("lightgbm_crypto")
class CryptoPipeline(LightGBMPipeline):
    """LightGBM pipeline for Numerai Crypto competition."""

    @classmethod
    def get_default_config(cls) -> PipelineConfig:
        """Get default configuration for Crypto pipeline."""
        config = super().get_default_config()
        config.name = "lightgbm_crypto"
        config.competition = CompetitionType.CRYPTO
        config.description = "LightGBM baseline for Numerai Crypto"

        # Crypto-specific parameters
        config.model_params.update(
            {
                "n_estimators": 1000,  # Fewer estimators for faster crypto predictions
                "learning_rate": 0.02,
                "max_depth": 4,
            }
        )

        return config
