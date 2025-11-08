"""Pipeline abstraction for training workflows."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import pandas as pd
import structlog

from .config import PipelineConfig, RunConfig


logger = structlog.get_logger()


class Pipeline(ABC):
    """Abstract base class for training pipelines.
    
    Subclasses should implement:
    - train(): Main training logic
    - predict(): Generate predictions
    - load_data(): Load and prepare data
    - get_default_config(): Return default configuration
    """

    def __init__(self, config: PipelineConfig, run_config: RunConfig):
        self.config = config
        self.run_config = run_config
        self.logger = logger.bind(
            pipeline=config.name,
            run_id=run_config.run_id,
            competition=config.competition.value,
        )
        self._callbacks: List[Callable] = []
        self.model = None
        self.training_data = None
        self.validation_data = None

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """Execute the training pipeline.
        
        Returns:
            Dict containing training metrics and results
        """
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for the given data.
        
        Args:
            data: Input features
            
        Returns:
            DataFrame with predictions
        """
        pass

    @abstractmethod
    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load and prepare training and validation data.
        
        Returns:
            Tuple of (training_data, validation_data)
        """
        pass

    @classmethod
    @abstractmethod
    def get_default_config(cls) -> PipelineConfig:
        """Get the default configuration for this pipeline.
        
        Returns:
            Default PipelineConfig
        """
        pass

    def add_callback(self, callback: Callable) -> None:
        """Add a callback function to be called during training.
        
        Callbacks receive (event_type: str, data: Dict[str, Any])
        """
        self._callbacks.append(callback)

    def emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event to all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                self.logger.error("callback_error", error=str(e), event_type=event_type)

    def save_model(self, path: Optional[Path] = None) -> Path:
        """Save the trained model.
        
        Args:
            path: Optional path to save to, otherwise uses run_config
            
        Returns:
            Path where model was saved
        """
        if path is None:
            path = self.run_config.run_dir / "model.pkl"
        
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        
        self.logger.info("model_saved", path=str(path))
        return path

    def load_model(self, path: Path) -> None:
        """Load a trained model from disk.
        
        Args:
            path: Path to the saved model
        """
        import pickle
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        
        self.logger.info("model_loaded", path=str(path))

    def save_predictions(self, predictions: pd.DataFrame, path: Optional[Path] = None) -> Path:
        """Save predictions to disk.
        
        Args:
            predictions: DataFrame with predictions
            path: Optional path to save to
            
        Returns:
            Path where predictions were saved
        """
        if path is None:
            path = self.run_config.run_dir / "predictions.csv"
        
        predictions.to_csv(path, index=True)
        self.logger.info("predictions_saved", path=str(path), rows=len(predictions))
        return path

    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save metrics to the run configuration.
        
        Args:
            metrics: Dictionary of metrics to save
        """
        self.run_config.metrics.update(metrics)
        
        # Save to disk
        metrics_path = self.run_config.run_dir / "metrics" / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.run_config.metrics, f, indent=2, default=str)
        
        self.logger.info("metrics_saved", metrics=metrics)


class PipelineRegistry:
    """Registry for managing available pipelines."""

    _pipelines: Dict[str, Type[Pipeline]] = {}

    @classmethod
    def register(cls, name: str, pipeline_class: Type[Pipeline]) -> None:
        """Register a pipeline class.
        
        Args:
            name: Unique name for the pipeline
            pipeline_class: Pipeline class to register
        """
        cls._pipelines[name] = pipeline_class
        logger.info("pipeline_registered", name=name, class_name=pipeline_class.__name__)

    @classmethod
    def get(cls, name: str) -> Type[Pipeline]:
        """Get a registered pipeline class.
        
        Args:
            name: Name of the pipeline
            
        Returns:
            Pipeline class
            
        Raises:
            KeyError: If pipeline not found
        """
        return cls._pipelines[name]

    @classmethod
    def list_pipelines(cls) -> List[str]:
        """List all registered pipeline names.
        
        Returns:
            List of pipeline names
        """
        return list(cls._pipelines.keys())

    @classmethod
    def get_pipeline_info(cls, name: str) -> Dict[str, Any]:
        """Get information about a pipeline.
        
        Args:
            name: Name of the pipeline
            
        Returns:
            Dictionary with pipeline information
        """
        pipeline_class = cls._pipelines[name]
        default_config = pipeline_class.get_default_config()
        
        return {
            "name": name,
            "class": pipeline_class.__name__,
            "competition": default_config.competition.value,
            "version": default_config.version,
            "description": default_config.description,
        }


def register_pipeline(name: str):
    """Decorator to register a pipeline class.
    
    Usage:
        @register_pipeline("my_pipeline")
        class MyPipeline(Pipeline):
            ...
    """
    def decorator(cls: Type[Pipeline]):
        PipelineRegistry.register(name, cls)
        return cls
    return decorator
