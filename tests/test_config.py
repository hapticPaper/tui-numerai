"""Tests for configuration models."""

from pathlib import Path
from datetime import datetime

import pytest

from tui_numerai.core.config import (
    CompetitionType,
    CompetitionConfig,
    PipelineConfig,
    RunConfig,
)


def test_competition_config():
    """Test CompetitionConfig creation."""
    config = CompetitionConfig(
        competition=CompetitionType.NUMERAI,
        data_dir=Path("/tmp/data"),
        models_dir=Path("/tmp/models"),
        predictions_dir=Path("/tmp/predictions"),
    )
    
    assert config.competition == CompetitionType.NUMERAI
    assert config.data_dir == Path("/tmp/data")
    assert config.models_dir == Path("/tmp/models")
    assert config.predictions_dir == Path("/tmp/predictions")


def test_pipeline_config():
    """Test PipelineConfig creation."""
    config = PipelineConfig(
        name="test_pipeline",
        competition=CompetitionType.NUMERAI,
        version="1.0.0",
        description="Test pipeline",
        model_params={"n_estimators": 100},
    )
    
    assert config.name == "test_pipeline"
    assert config.competition == CompetitionType.NUMERAI
    assert config.version == "1.0.0"
    assert config.model_params["n_estimators"] == 100


def test_run_config():
    """Test RunConfig creation."""
    pipeline_config = PipelineConfig(
        name="test_pipeline",
        competition=CompetitionType.NUMERAI,
    )
    
    config = RunConfig(
        run_id="test_run_123",
        pipeline_name="test_pipeline",
        pipeline_version="1.0.0",
        competition=CompetitionType.NUMERAI,
        pipeline_config=pipeline_config,
        run_dir=Path("/tmp/runs/test_run_123"),
    )
    
    assert config.run_id == "test_run_123"
    assert config.status == "created"
    assert config.pipeline_name == "test_pipeline"
