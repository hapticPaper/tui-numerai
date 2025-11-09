"""Test script for LightGBM pipeline."""

from pathlib import Path
import tempfile

from tui_numerai.core import PipelineRegistry, RunConfig
from tui_numerai.core.config import CompetitionType
from tui_numerai.utils.logging import setup_logging

# Import pipelines to register them
from tui_numerai.pipelines import LightGBMPipeline

# Setup logging
setup_logging(log_level="INFO")

# Get pipeline
pipeline_class = PipelineRegistry.get("lightgbm_numerai")
pipeline_config = pipeline_class.get_default_config()

# Create a temporary directory for the run
with tempfile.TemporaryDirectory() as tmpdir:
    run_dir = Path(tmpdir) / "test_run"
    
    run_config = RunConfig(
        run_id="test_run",
        pipeline_name="lightgbm_numerai",
        pipeline_version="1.0.0",
        competition=CompetitionType.NUMERAI,
        pipeline_config=pipeline_config,
        run_dir=run_dir,
    )
    
    run_config.ensure_run_directory()
    
    # Create pipeline instance
    pipeline = pipeline_class(pipeline_config, run_config)
    
    # Add a simple callback to print events
    def on_event(event_type: str, data: dict) -> None:
        if event_type == "log":
            print(f"[LOG] {data.get('message', '')}")
        elif event_type == "metrics":
            print(f"[METRICS] {data}")
    
    pipeline.add_callback(on_event)
    
    # Run training
    print("\n=== Starting Training ===\n")
    results = pipeline.train()
    
    print("\n=== Training Complete ===")
    print(f"Results: {results}")
