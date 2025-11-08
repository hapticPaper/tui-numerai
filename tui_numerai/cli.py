"""Command-line interface for TUI Numerai."""

import argparse
from pathlib import Path

from .tui.app import NumeraiTUI
from .utils.logging import setup_logging

# Import pipelines to register them
from .pipelines import (
    LightGBMPipeline,
    PyTorchPipeline,
    LightGBMWandbPipeline,
    CryptoPipeline,
    SignalsPipeline,
)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="TUI Numerai - Terminal UI for Numerai training"
    )
    
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("./runs"),
        help="Directory to store run data (default: ./runs)",
    )
    
    parser.add_argument(
        "--competition",
        type=str,
        choices=["numerai", "crypto", "signals"],
        help="Competition type (optional, can select in UI)",
    )
    
    parser.add_argument(
        "--pipeline",
        type=str,
        help="Pipeline name (optional, can select in UI)",
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Optional log file path",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        log_level=args.log_level,
        log_file=args.log_file,
    )
    
    # Create and run the TUI app
    app = NumeraiTUI(runs_dir=args.runs_dir)
    
    # If specific pipeline was requested, we could pre-select it here
    # For now, we'll let users select via the UI
    
    app.run()


if __name__ == "__main__":
    main()
