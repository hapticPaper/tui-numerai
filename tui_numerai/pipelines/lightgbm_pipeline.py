"""LightGBM pipeline for Numerai."""

from pathlib import Path
from typing import Any, Dict

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..core import Pipeline, PipelineConfig, RunConfig
from ..core.config import CompetitionType
from ..core.pipeline import register_pipeline


@register_pipeline("lightgbm_numerai")
class LightGBMPipeline(Pipeline):
    """LightGBM pipeline for Numerai competition."""

    def __init__(self, config: PipelineConfig, run_config: RunConfig):
        super().__init__(config, run_config)
        self.feature_cols = None

    @classmethod
    def get_default_config(cls) -> PipelineConfig:
        """Get default configuration for LightGBM pipeline."""
        return PipelineConfig(
            name="lightgbm_numerai",
            competition=CompetitionType.NUMERAI,
            version="1.0.0",
            description="LightGBM baseline for Numerai Classic",
            model_params={
                "n_estimators": 2000,
                "learning_rate": 0.01,
                "max_depth": 5,
                "num_leaves": 32,
                "colsample_bytree": 0.1,
                "random_state": 42,
            },
            target_name="target",
            feature_set="all",
        )

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and validation data.
        
        This is a placeholder - in production, you'd download from Numerai API.
        """
        self.emit_event("log", {"message": "Loading data..."})
        
        # In a real implementation, you would use numerapi to download data
        # For now, we'll create placeholder dataframes
        # napi = numerapi.NumerAPI()
        # napi.download_dataset("v5.0/train_int8.parquet")
        
        # Create dummy data for demonstration
        n_samples = 1000
        n_features = 20
        
        self.feature_cols = [f"feature_{i}" for i in range(n_features)]
        
        # Generate random data
        np.random.seed(42)
        train_features = np.random.randn(n_samples, n_features)
        train_target = np.random.rand(n_samples)
        
        train_data = pd.DataFrame(train_features, columns=self.feature_cols)
        train_data["era"] = np.repeat(np.arange(n_samples // 100), 100)
        train_data[self.config.target_name] = train_target
        
        val_size = n_samples // 4
        val_features = np.random.randn(val_size, n_features)
        val_target = np.random.rand(val_size)
        
        val_data = pd.DataFrame(val_features, columns=self.feature_cols)
        n_val_eras = max(1, val_size // 100)
        val_data["era"] = np.repeat(np.arange(n_val_eras), val_size // n_val_eras)[:val_size]
        val_data[self.config.target_name] = val_target
        
        self.emit_event("log", {
            "message": f"Loaded {len(train_data)} training samples, {len(val_data)} validation samples"
        })
        
        return train_data, val_data

    def train(self) -> Dict[str, Any]:
        """Train the LightGBM model."""
        self.emit_event("log", {"message": "Starting LightGBM training..."})
        
        # Load data
        train_data, val_data = self.load_data()
        self.training_data = train_data
        self.validation_data = val_data
        
        # Prepare features and targets
        X_train = train_data[self.feature_cols]
        y_train = train_data[self.config.target_name]
        X_val = val_data[self.feature_cols]
        y_val = val_data[self.config.target_name]
        
        # Create datasets
        train_set = lgb.Dataset(X_train, y_train)
        val_set = lgb.Dataset(X_val, y_val, reference=train_set)
        
        # Training parameters
        params = {
            "objective": "regression",
            "metric": "l2",
            "verbosity": 1,
            **self.config.model_params
        }
        
        # Track metrics during training
        callbacks = [
            lgb.log_evaluation(period=100),
        ]
        
        # Train model
        self.emit_event("log", {"message": "Training model with LightGBM..."})
        
        self.model = lgb.train(
            params,
            train_set,
            valid_sets=[train_set, val_set],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )
        
        # Generate predictions
        train_preds = self.model.predict(X_train)
        val_preds = self.model.predict(X_val)
        
        # Calculate metrics
        train_corr = np.corrcoef(y_train, train_preds)[0, 1]
        val_corr = np.corrcoef(y_val, val_preds)[0, 1]
        
        metrics = {
            "train_correlation": float(train_corr),
            "val_correlation": float(val_corr),
            "n_estimators": self.model.num_trees(),
        }
        
        self.emit_event("metrics", metrics)
        self.save_metrics(metrics)
        
        # Save model
        if self.config.save_model:
            model_path = self.save_model()
            self.emit_event("log", {"message": f"Model saved to {model_path}"})
        
        # Save predictions
        if self.config.save_predictions:
            val_data["prediction"] = val_preds
            pred_path = self.save_predictions(val_data[["prediction"]])
            self.emit_event("log", {"message": f"Predictions saved to {pred_path}"})
        
        self.emit_event("log", {"message": "Training complete!"})
        
        return metrics

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for new data."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X = data[self.feature_cols]
        predictions = self.model.predict(X)
        
        result = pd.DataFrame({"prediction": predictions}, index=data.index)
        return result
