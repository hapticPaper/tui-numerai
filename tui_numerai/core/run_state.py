"""Run state management for resuming training."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from pydantic import BaseModel, ConfigDict

from .config import RunConfig


logger = structlog.get_logger()


class RunState(BaseModel):
    """State information for a training run."""

    run_id: str
    epoch: int = 0
    step: int = 0
    best_metric: Optional[float] = None
    best_epoch: Optional[int] = None
    checkpoint_path: Optional[str] = None
    custom_state: Dict[str, Any] = {}

    model_config = ConfigDict(extra="allow")


class RunStateManager:
    """Manage run states for resuming training."""

    def __init__(self, base_dir: Path = Path("./runs")):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_run_config(self, run_config: RunConfig) -> None:
        """Save run configuration to disk.
        
        Args:
            run_config: Configuration to save
        """
        config_path = run_config.run_dir / "run_config.json"
        with open(config_path, "w") as f:
            json.dump(run_config.model_dump(mode="json"), f, indent=2, default=str)
        
        logger.info("run_config_saved", run_id=run_config.run_id, path=str(config_path))

    def load_run_config(self, run_id: str) -> RunConfig:
        """Load run configuration from disk.
        
        Args:
            run_id: ID of the run to load
            
        Returns:
            Loaded RunConfig
        """
        run_dir = self.base_dir / run_id
        config_path = run_dir / "run_config.json"
        
        with open(config_path, "r") as f:
            config_data = json.load(f)
        
        # Convert string paths back to Path objects
        for key in ["run_dir", "model_path", "predictions_path"]:
            if config_data.get(key):
                config_data[key] = Path(config_data[key])
        
        logger.info("run_config_loaded", run_id=run_id)
        return RunConfig(**config_data)

    def save_state(self, run_id: str, state: RunState) -> None:
        """Save run state to disk.
        
        Args:
            run_id: ID of the run
            state: State to save
        """
        run_dir = self.base_dir / run_id
        state_path = run_dir / "run_state.json"
        
        with open(state_path, "w") as f:
            json.dump(state.model_dump(mode="json"), f, indent=2, default=str)
        
        logger.info("run_state_saved", run_id=run_id, epoch=state.epoch)

    def load_state(self, run_id: str) -> Optional[RunState]:
        """Load run state from disk.
        
        Args:
            run_id: ID of the run
            
        Returns:
            Loaded RunState or None if not found
        """
        run_dir = self.base_dir / run_id
        state_path = run_dir / "run_state.json"
        
        if not state_path.exists():
            logger.warning("run_state_not_found", run_id=run_id)
            return None
        
        with open(state_path, "r") as f:
            state_data = json.load(f)
        
        logger.info("run_state_loaded", run_id=run_id)
        return RunState(**state_data)

    def list_runs(self, competition: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all available runs.
        
        Args:
            competition: Optional filter by competition type
            
        Returns:
            List of run information dictionaries
        """
        runs = []
        
        for run_dir in self.base_dir.iterdir():
            if not run_dir.is_dir():
                continue
            
            config_path = run_dir / "run_config.json"
            if not config_path.exists():
                continue
            
            try:
                with open(config_path, "r") as f:
                    config_data = json.load(f)
                
                if competition and config_data.get("competition") != competition:
                    continue
                
                runs.append({
                    "run_id": config_data["run_id"],
                    "pipeline": config_data["pipeline_name"],
                    "competition": config_data["competition"],
                    "status": config_data.get("status", "unknown"),
                    "created_at": config_data.get("created_at"),
                    "completed_at": config_data.get("completed_at"),
                })
            except Exception as e:
                logger.error("error_loading_run", run_id=run_dir.name, error=str(e))
        
        # Sort by creation time
        runs.sort(key=lambda x: x["created_at"], reverse=True)
        return runs

    def get_resumable_runs(self, pipeline_name: str) -> List[Dict[str, Any]]:
        """Get runs that can be resumed for a specific pipeline.
        
        Args:
            pipeline_name: Name of the pipeline
            
        Returns:
            List of resumable run information
        """
        all_runs = self.list_runs()
        resumable = []
        
        for run_info in all_runs:
            if run_info["pipeline"] != pipeline_name:
                continue
            
            if run_info["status"] in ["running", "failed"]:
                resumable.append(run_info)
        
        return resumable

    def create_run_id(self, pipeline_name: str) -> str:
        """Create a new unique run ID.
        
        Args:
            pipeline_name: Name of the pipeline
            
        Returns:
            New run ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{pipeline_name}_{timestamp}"
        return run_id

    def get_run_directory(self, run_id: str) -> Path:
        """Get the directory path for a run.
        
        Args:
            run_id: ID of the run
            
        Returns:
            Path to the run directory
        """
        return self.base_dir / run_id
