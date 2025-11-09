# TUI Numerai Usage Guide

## Installation

### Basic Installation

```bash
cd tui-numerai
pip install -e .
```

### With Optional Dependencies

```bash
# For Weights & Biases integration
pip install -e ".[wandb]"

# For InfluxDB integration
pip install -e ".[influx]"

# For development
pip install -e ".[dev]"

# All optional dependencies
pip install -e ".[wandb,influx,dev]"
```

## Quick Start

### Launch the TUI

The simplest way to start:

```bash
tui-numerai
```

This will launch the interactive TUI where you can:
1. Select a competition and pipeline
2. Configure parameters
3. Start training
4. Monitor progress in real-time

### Command-Line Options

```bash
# Specify runs directory
tui-numerai --runs-dir ./my_runs

# Set log level
tui-numerai --log-level DEBUG

# Save logs to file
tui-numerai --log-file training.log
```

## Using the TUI

### Main Menu

When you launch the TUI, you'll see a list of available pipelines:

- **lightgbm_numerai**: LightGBM baseline for Numerai Classic
- **pytorch_numerai**: PyTorch neural network for Numerai Classic
- **lightgbm_numerai_wandb**: LightGBM with W&B integration
- **lightgbm_crypto**: LightGBM for Crypto competition
- **lightgbm_signals**: LightGBM for Signals competition

Use arrow keys to navigate and Enter to select.

### Run Selection

After selecting a pipeline, you can:
- **Resume a previous run**: Select from failed or interrupted runs
- **Start a new run**: Create a fresh training run

### Training Screen

The training screen shows:

**Left Panel - Output Log**:
- Real-time training output
- Library logs (e.g., LightGBM progress)
- Event messages

**Right Panel - Metrics**:
- Live training metrics
- Validation scores
- Model parameters

**Bottom Buttons**:
- Stop Training: Interrupt the current run
- Back to Menu: Return to pipeline selection

### Keyboard Shortcuts

- `q`: Quit the application
- `s`: Stop current training
- `Esc`: Go back to previous screen
- Arrow keys: Navigate lists and tables
- Enter: Select/confirm

## Programmatic Usage

You can also use the framework programmatically without the TUI.

### Example: Train a Pipeline

```python
from pathlib import Path
from tui_numerai.core import PipelineRegistry, RunConfig
from tui_numerai.core.config import CompetitionType
from tui_numerai.pipelines import LightGBMPipeline

# Get pipeline class
pipeline_class = PipelineRegistry.get("lightgbm_numerai")
pipeline_config = pipeline_class.get_default_config()

# Configure the pipeline
pipeline_config.model_params["n_estimators"] = 1000
pipeline_config.model_params["learning_rate"] = 0.02

# Create run configuration
run_config = RunConfig(
    run_id="my_experiment_001",
    pipeline_name="lightgbm_numerai",
    pipeline_version="1.0.0",
    competition=CompetitionType.NUMERAI,
    pipeline_config=pipeline_config,
    run_dir=Path("./runs/my_experiment_001"),
)

run_config.ensure_run_directory()

# Create pipeline instance
pipeline = pipeline_class(pipeline_config, run_config)

# Add callback for logging
def log_callback(event_type, data):
    if event_type == "log":
        print(data.get("message"))
    elif event_type == "metrics":
        print(f"Metrics: {data}")

pipeline.add_callback(log_callback)

# Run training
results = pipeline.train()
print(f"Training complete! Results: {results}")
```

### Example: Resume a Run

```python
from tui_numerai.core import RunStateManager

# Load previous run
manager = RunStateManager(Path("./runs"))
run_config = manager.load_run_config("my_experiment_001")
state = manager.load_state("my_experiment_001")

# Resume training from checkpoint
pipeline = pipeline_class(run_config.pipeline_config, run_config)
# ... load checkpoint and continue training
```

### Example: Custom Pipeline

```python
from tui_numerai.core import Pipeline, PipelineConfig
from tui_numerai.core.pipeline import register_pipeline

@register_pipeline("my_custom_pipeline")
class MyCustomPipeline(Pipeline):
    
    @classmethod
    def get_default_config(cls):
        return PipelineConfig(
            name="my_custom_pipeline",
            competition=CompetitionType.NUMERAI,
            description="My custom training pipeline",
        )
    
    def load_data(self):
        # Load your data
        return train_data, val_data
    
    def train(self):
        # Your training logic
        self.emit_event("log", {"message": "Starting training"})
        # ... training code ...
        return {"accuracy": 0.95}
    
    def predict(self, data):
        # Your prediction logic
        return predictions
```

## Configuration

### Pipeline Configuration

Each pipeline has default configuration that can be customized:

```python
config = pipeline_class.get_default_config()

# Model parameters
config.model_params["n_estimators"] = 2000
config.model_params["learning_rate"] = 0.01

# Training parameters
config.epochs = 100
config.batch_size = 512
config.learning_rate = 0.001

# Feature configuration
config.feature_set = "medium"
config.neutralize_features = True

# Output configuration
config.save_predictions = True
config.save_model = True

# Logging configuration
config.use_wandb = True
config.wandb_project = "numerai-experiments"
```

### Run Directory Structure

Each run creates a structured directory:

```
runs/
  └── lightgbm_numerai_20240101_120000/
      ├── run_config.json       # Full run configuration
      ├── run_state.json        # Checkpoint state
      ├── model.pkl             # Saved model
      ├── predictions.csv       # Predictions
      ├── checkpoints/
      │   └── best_model.pt     # Best checkpoint
      ├── logs/
      │   └── training.log      # Detailed logs
      ├── metrics/
      │   └── metrics.json      # Metrics history
      └── plots/
          └── correlation.png   # Visualization plots
```

## Integration Examples

### Weights & Biases

```python
pipeline_config.use_wandb = True
pipeline_config.wandb_project = "numerai-classic"

# W&B will automatically track:
# - Model configuration
# - Training metrics
# - Validation scores
```

### InfluxDB

```python
from tui_numerai.utils.influxdb import create_influxdb_reporter

reporter = create_influxdb_reporter(
    url="http://localhost:8086",
    token=os.getenv("INFLUXDB_TOKEN"),
    org="my-org",
    bucket="numerai-metrics",
)

# Report metrics to InfluxDB
reporter.report_metrics(
    metrics={"val_correlation": 0.025},
    tags={"pipeline": "lightgbm_numerai", "run_id": "exp_001"},
)
```

### Structured Logging

```python
from tui_numerai.utils.logging import setup_logging

# Setup logging with JSON output
setup_logging(
    log_level="INFO",
    log_file=Path("./logs/training.log"),
    use_json=True,  # JSON format for log aggregation
)
```

## Troubleshooting

### Pipeline Not Found

Make sure to import the pipeline to register it:
```python
from tui_numerai.pipelines import LightGBMPipeline
```

### Run Directory Permissions

Ensure the runs directory is writable:
```bash
mkdir -p ./runs
chmod 755 ./runs
```

### Memory Issues with Large Datasets

Use data streaming or chunking:
```python
def load_data(self):
    # Load in chunks
    for chunk in pd.read_csv("data.csv", chunksize=10000):
        yield chunk
```

### GPU Not Detected (PyTorch)

Check CUDA availability:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
```

## Advanced Topics

### Custom Callbacks

Add custom callbacks for advanced monitoring:

```python
def custom_callback(event_type, data):
    if event_type == "metrics":
        # Send to monitoring system
        send_to_monitoring(data)
    elif event_type == "checkpoint":
        # Upload checkpoint to cloud storage
        upload_checkpoint(data["path"])

pipeline.add_callback(custom_callback)
```

### Parallel Training

Run multiple pipelines in parallel:

```python
from concurrent.futures import ProcessPoolExecutor

def train_pipeline(config):
    pipeline = create_pipeline(config)
    return pipeline.train()

with ProcessPoolExecutor() as executor:
    futures = [executor.submit(train_pipeline, cfg) for cfg in configs]
    results = [f.result() for f in futures]
```

### Custom Data Loaders

Implement custom data loading:

```python
class MyPipeline(Pipeline):
    def load_data(self):
        # Download from Numerai API
        import numerapi
        napi = numerapi.NumerAPI()
        
        # Download latest data
        napi.download_dataset("v5.0/train.parquet", "data/train.parquet")
        
        # Load and process
        train = pd.read_parquet("data/train.parquet")
        val = pd.read_parquet("data/val.parquet")
        
        return train, val
```
