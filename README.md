# TUI-Numerai

A powerful Terminal User Interface (TUI) framework for Numerai training and validation, built with [Textual](https://textual.textualize.io/).

## Features

- **Multi-Competition Support**: Seamlessly work with Numerai Classic, Crypto, and Signals competitions
- **Pipeline Versioning**: Create and manage different pipeline variations (LightGBM, PyTorch, with/without WandB, etc.)
- **Pydantic Configuration**: Type-safe, validated configuration management
- **Run Resumption**: Resume training from previous runs with full state restoration
- **Real-time Monitoring**: Live output capture, metrics display, and image viewing
- **Structured Logging**: Built-in structlog integration with optional InfluxDB export
- **Reusable Components**: Highly modular TUI components for custom workflows

## Installation

```bash
pip install -e .

# With optional dependencies
pip install -e ".[wandb,influx,dev]"
```

## Quick Start

```bash
# Launch the TUI
tui-numerai

# Or run a specific pipeline
tui-numerai --competition numerai --pipeline lightgbm
```

## Architecture

The framework provides a clean abstraction layer:

1. **Pipeline Base**: Abstract base class for defining training pipelines
2. **Config Management**: Pydantic models for type-safe configuration
3. **Run State**: Persistent state management for resuming training
4. **TUI Components**: Reusable widgets for output capture, metrics, and visualization

## Example Pipelines

The framework includes implementations based on the [Numerai example-scripts](https://github.com/numerai/example-scripts):

- LightGBM baseline
- PyTorch neural network
- Feature neutralization
- Target ensemble
- WandB integration variants

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black .
ruff check .
```

## License

MIT
