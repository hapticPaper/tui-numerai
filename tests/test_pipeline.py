"""Tests for pipeline abstraction."""

from pathlib import Path
import tempfile

from tui_numerai.core import PipelineRegistry
from tui_numerai.core.config import CompetitionType, RunConfig
from tui_numerai.pipelines import LightGBMPipeline


def test_pipeline_registry():
    """Test pipeline registration and retrieval."""
    # Check that our pipelines are registered
    pipelines = PipelineRegistry.list_pipelines()

    assert "lightgbm_numerai" in pipelines
    assert "pytorch_numerai" in pipelines
    assert "lightgbm_numerai_wandb" in pipelines


def test_pipeline_info():
    """Test getting pipeline information."""
    info = PipelineRegistry.get_pipeline_info("lightgbm_numerai")

    assert info["name"] == "lightgbm_numerai"
    assert info["competition"] == "numerai"
    assert "version" in info
    assert "description" in info


def test_lightgbm_pipeline_config():
    """Test LightGBM pipeline default configuration."""
    config = LightGBMPipeline.get_default_config()

    assert config.name == "lightgbm_numerai"
    assert config.competition == CompetitionType.NUMERAI
    assert "n_estimators" in config.model_params


def test_lightgbm_pipeline_initialization():
    """Test LightGBM pipeline initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline_config = LightGBMPipeline.get_default_config()

        run_config = RunConfig(
            run_id="test_run",
            pipeline_name="lightgbm_numerai",
            pipeline_version="1.0.0",
            competition=CompetitionType.NUMERAI,
            pipeline_config=pipeline_config,
            run_dir=Path(tmpdir) / "test_run",
        )

        pipeline = LightGBMPipeline(pipeline_config, run_config)

        assert pipeline.config == pipeline_config
        assert pipeline.run_config == run_config


def test_pipeline_callbacks():
    """Test pipeline event callbacks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline_config = LightGBMPipeline.get_default_config()

        run_config = RunConfig(
            run_id="test_run",
            pipeline_name="lightgbm_numerai",
            pipeline_version="1.0.0",
            competition=CompetitionType.NUMERAI,
            pipeline_config=pipeline_config,
            run_dir=Path(tmpdir) / "test_run",
        )

        pipeline = LightGBMPipeline(pipeline_config, run_config)

        events_received = []

        def callback(event_type, data):
            events_received.append((event_type, data))

        pipeline.add_callback(callback)
        pipeline.emit_event("test_event", {"key": "value"})

        assert len(events_received) == 1
        assert events_received[0][0] == "test_event"
        assert events_received[0][1]["key"] == "value"
