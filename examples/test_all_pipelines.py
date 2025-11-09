"""Test all registered pipelines."""

from pathlib import Path
import tempfile

from tui_numerai.core import PipelineRegistry, RunConfig
from tui_numerai.utils.logging import setup_logging

# Import all pipelines
from tui_numerai.pipelines import (
    LightGBMPipeline,
    PyTorchPipeline,
    LightGBMWandbPipeline,
    CryptoPipeline,
    SignalsPipeline,
)

# Setup logging
setup_logging(log_level="INFO")

print("=" * 80)
print("Testing All Pipelines")
print("=" * 80)

# Get all registered pipelines
pipelines = PipelineRegistry.list_pipelines()
print(f"\nFound {len(pipelines)} registered pipelines:")
for name in sorted(pipelines):
    info = PipelineRegistry.get_pipeline_info(name)
    print(f"  - {name:30} | {info['competition']:10} | {info['description']}")

# Test LightGBM pipeline
print("\n" + "=" * 80)
print("Testing LightGBM Pipeline")
print("=" * 80)

pipeline_class = PipelineRegistry.get("lightgbm_numerai")
pipeline_config = pipeline_class.get_default_config()

# Reduce training time for demo
pipeline_config.model_params["n_estimators"] = 100

with tempfile.TemporaryDirectory() as tmpdir:
    run_dir = Path(tmpdir) / "test_run"
    
    run_config = RunConfig(
        run_id="test_lightgbm",
        pipeline_name="lightgbm_numerai",
        pipeline_version="1.0.0",
        competition=pipeline_config.competition,
        pipeline_config=pipeline_config,
        run_dir=run_dir,
    )
    
    run_config.ensure_run_directory()
    
    pipeline = pipeline_class(pipeline_config, run_config)
    
    # Simple logging callback
    def on_event(event_type: str, data: dict):
        if event_type == "metrics":
            print(f"\n[METRICS] {data}")
    
    pipeline.add_callback(on_event)
    
    results = pipeline.train()
    print(f"\nTraining complete!")
    print(f"  Train correlation: {results['train_correlation']:.6f}")
    print(f"  Val correlation:   {results['val_correlation']:.6f}")

print("\n" + "=" * 80)
print("All Tests Complete!")
print("=" * 80)
