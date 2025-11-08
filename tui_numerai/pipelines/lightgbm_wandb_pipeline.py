"""LightGBM pipeline with Weights & Biases integration."""

from typing import Any, Dict

from ..core import PipelineConfig
from ..core.config import CompetitionType
from ..core.pipeline import register_pipeline
from .lightgbm_pipeline import LightGBMPipeline


@register_pipeline("lightgbm_numerai_wandb")
class LightGBMWandbPipeline(LightGBMPipeline):
    """LightGBM pipeline with W&B logging."""

    @classmethod
    def get_default_config(cls) -> PipelineConfig:
        """Get default configuration with W&B enabled."""
        config = super().get_default_config()
        config.name = "lightgbm_numerai_wandb"
        config.description = "LightGBM with Weights & Biases logging"
        config.use_wandb = True
        config.wandb_project = "numerai-training"
        return config

    def train(self) -> Dict[str, Any]:
        """Train with W&B logging."""
        # Initialize W&B if configured
        if self.config.use_wandb:
            try:
                import wandb
                
                wandb.init(
                    project=self.config.wandb_project,
                    name=self.run_config.run_id,
                    config=self.config.model_dump(),
                )
                
                self.emit_event("log", {"message": "Initialized Weights & Biases"})
            except ImportError:
                self.emit_event("log", {
                    "message": "Warning: wandb not installed, skipping W&B logging"
                })
        
        # Run training
        metrics = super().train()
        
        # Log to W&B
        if self.config.use_wandb:
            try:
                import wandb
                wandb.log(metrics)
                wandb.finish()
            except:
                pass
        
        return metrics
