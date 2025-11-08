"""Tests for run state management."""

import tempfile
from pathlib import Path

import pytest

from tui_numerai.core import RunStateManager, RunState
from tui_numerai.core.config import CompetitionType, PipelineConfig, RunConfig


def test_run_state_manager():
    """Test RunStateManager basic functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = RunStateManager(Path(tmpdir))
        
        assert manager.base_dir == Path(tmpdir)
        assert manager.base_dir.exists()


def test_save_and_load_run_config():
    """Test saving and loading run configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = RunStateManager(Path(tmpdir))
        
        pipeline_config = PipelineConfig(
            name="test_pipeline",
            competition=CompetitionType.NUMERAI,
        )
        
        run_config = RunConfig(
            run_id="test_run_123",
            pipeline_name="test_pipeline",
            pipeline_version="1.0.0",
            competition=CompetitionType.NUMERAI,
            pipeline_config=pipeline_config,
            run_dir=Path(tmpdir) / "test_run_123",
        )
        
        run_config.ensure_run_directory()
        manager.save_run_config(run_config)
        
        # Load it back
        loaded_config = manager.load_run_config("test_run_123")
        
        assert loaded_config.run_id == run_config.run_id
        assert loaded_config.pipeline_name == run_config.pipeline_name
        assert loaded_config.status == run_config.status


def test_save_and_load_state():
    """Test saving and loading run state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = RunStateManager(Path(tmpdir))
        
        run_dir = Path(tmpdir) / "test_run"
        run_dir.mkdir()
        
        state = RunState(
            run_id="test_run",
            epoch=10,
            step=1000,
            best_metric=0.95,
            best_epoch=8,
        )
        
        manager.save_state("test_run", state)
        
        # Load it back
        loaded_state = manager.load_state("test_run")
        
        assert loaded_state is not None
        assert loaded_state.run_id == state.run_id
        assert loaded_state.epoch == state.epoch
        assert loaded_state.best_metric == state.best_metric


def test_create_run_id():
    """Test creating unique run IDs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = RunStateManager(Path(tmpdir))
        
        run_id = manager.create_run_id("test_pipeline")
        
        assert "test_pipeline" in run_id
        assert len(run_id) > len("test_pipeline")


def test_list_runs():
    """Test listing available runs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = RunStateManager(Path(tmpdir))
        
        # Create a few runs
        for i in range(3):
            pipeline_config = PipelineConfig(
                name="test_pipeline",
                competition=CompetitionType.NUMERAI,
            )
            
            run_config = RunConfig(
                run_id=f"test_run_{i}",
                pipeline_name="test_pipeline",
                pipeline_version="1.0.0",
                competition=CompetitionType.NUMERAI,
                pipeline_config=pipeline_config,
                run_dir=Path(tmpdir) / f"test_run_{i}",
            )
            
            run_config.ensure_run_directory()
            manager.save_run_config(run_config)
        
        # List all runs
        runs = manager.list_runs()
        
        assert len(runs) == 3
        assert all("test_run" in run["run_id"] for run in runs)
