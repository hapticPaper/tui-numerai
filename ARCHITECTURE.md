# TUI Numerai Architecture

This document describes the architecture and design principles of the TUI Numerai framework.

## Overview

TUI Numerai is a Terminal User Interface (TUI) framework for training and validating models for Numerai competitions (Classic, Crypto, Signals). It provides a clean abstraction layer that makes it easy to:

1. Define new training pipelines
2. Configure and run experiments
3. Monitor training progress in real-time
4. Resume interrupted training runs
5. Manage multiple pipeline versions

## Core Design Principles

### 1. Well-Abstracted Pipeline System

The `Pipeline` abstract base class defines the interface that all training pipelines must implement:

```python
class Pipeline(ABC):
    def train(self) -> Dict[str, Any]: ...
    def predict(self, data: pd.DataFrame) -> pd.DataFrame: ...
    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]: ...
    @classmethod
    def get_default_config(cls) -> PipelineConfig: ...
```

This abstraction allows easy creation of new pipelines without modifying the TUI code.

### 2. Type-Safe Configuration with Pydantic

All configuration is managed through Pydantic models:

- `PipelineConfig`: Pipeline-specific configuration (model params, features, etc.)
- `RunConfig`: Run-specific configuration (paths, metadata, status)
- `CompetitionConfig`: Competition-specific settings

Pydantic provides:
- Type validation
- Serialization/deserialization
- Default values
- Documentation

### 3. Event-Driven Updates

Pipelines emit events that the TUI listens to:

```python
pipeline.emit_event("log", {"message": "Training started"})
pipeline.emit_event("metrics", {"loss": 0.5, "accuracy": 0.9})
```

This decouples the training logic from the UI, making it easy to:
- Test pipelines without the UI
- Use different frontends (CLI, web, etc.)
- Capture and log all training events

### 4. Persistent State Management

The `RunStateManager` handles:
- Saving/loading run configurations
- Saving/loading checkpoints for resumption
- Listing and filtering past runs
- Creating unique run IDs

All run data is stored in a structured directory:
```
runs/
  └── pipeline_name_20240101_120000/
      ├── run_config.json
      ├── run_state.json
      ├── checkpoints/
      ├── logs/
      ├── metrics/
      └── plots/
```

### 5. Reusable TUI Components

The framework provides reusable widgets:

- `OutputCapture`: Scrolling log viewer
- `MetricsDisplay`: Real-time metrics table
- `ParameterEditor`: Form for editing pipeline parameters
- `PipelineSelector`: List of available pipelines
- `RunSelector`: List of resumable runs

## Architecture Layers

### Layer 1: Core Abstractions (`tui_numerai/core/`)

- `config.py`: Pydantic configuration models
- `pipeline.py`: Pipeline base class and registry
- `run_state.py`: Run state management

### Layer 2: Pipeline Implementations (`tui_numerai/pipelines/`)

- `lightgbm_pipeline.py`: LightGBM baseline
- `pytorch_pipeline.py`: PyTorch neural network
- `lightgbm_wandb_pipeline.py`: LightGBM with W&B logging
- `crypto_pipeline.py`: Crypto competition variant
- `signals_pipeline.py`: Signals competition variant

### Layer 3: TUI Components (`tui_numerai/tui/`)

- `widgets.py`: Reusable UI components
- `app.py`: Main TUI application and screens

### Layer 4: Utilities (`tui_numerai/utils/`)

- `logging.py`: Structured logging setup
- `influxdb.py`: InfluxDB metrics reporter

### Layer 5: CLI (`tui_numerai/cli.py`)

Command-line interface that ties everything together.

## Creating a New Pipeline

Here's how to create a new pipeline:

```python
from tui_numerai.core import Pipeline, PipelineConfig
from tui_numerai.core.config import CompetitionType
from tui_numerai.core.pipeline import register_pipeline

@register_pipeline("my_pipeline")
class MyPipeline(Pipeline):
    
    @classmethod
    def get_default_config(cls) -> PipelineConfig:
        return PipelineConfig(
            name="my_pipeline",
            competition=CompetitionType.NUMERAI,
            version="1.0.0",
            description="My custom pipeline",
            model_params={"param1": 100},
        )
    
    def load_data(self):
        # Load and return training and validation data
        return train_data, val_data
    
    def train(self):
        # Implement training logic
        self.emit_event("log", {"message": "Training..."})
        # ... training code ...
        self.emit_event("metrics", {"accuracy": 0.9})
        return metrics
    
    def predict(self, data):
        # Generate predictions
        return predictions
```

The `@register_pipeline` decorator automatically registers the pipeline with the system.

## Integration Points

### Weights & Biases

To add W&B logging to a pipeline:

```python
config.use_wandb = True
config.wandb_project = "my-project"
```

The pipeline can then use:
```python
if self.config.use_wandb:
    import wandb
    wandb.init(project=self.config.wandb_project)
    wandb.log(metrics)
```

### InfluxDB

To report metrics to InfluxDB:

```python
from tui_numerai.utils.influxdb import create_influxdb_reporter

reporter = create_influxdb_reporter(url, token, org, bucket)
reporter.report_metrics(metrics, tags={"pipeline": "my_pipeline"})
```

### StructLog

All logging uses structlog for structured, contextual logs:

```python
self.logger.info("training_complete", 
                 accuracy=0.95, 
                 duration=120)
```

## Testing Strategy

The framework includes comprehensive tests:

1. **Configuration Tests**: Validate Pydantic models
2. **Pipeline Tests**: Test registration and callbacks
3. **State Management Tests**: Test save/load functionality

Run tests with:
```bash
pytest tests/
```

## Extension Points

The framework is designed to be extensible:

1. **Custom Pipelines**: Inherit from `Pipeline` and register
2. **Custom Widgets**: Inherit from Textual widgets
3. **Custom Screens**: Inherit from `Screen` and add to `SCREENS`
4. **Custom Callbacks**: Add callbacks to pipelines for custom behavior

## Performance Considerations

- Training runs in a separate thread pool to avoid blocking the UI
- Large datasets should be loaded lazily
- Metrics are aggregated before display to avoid UI lag
- Checkpoints are saved asynchronously

## Security Considerations

- Credentials should be stored in environment variables
- Run directories have restrictive permissions
- User input is validated through Pydantic
- Code execution is sandboxed within pipeline boundaries
