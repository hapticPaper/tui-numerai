# Contributing to TUI Numerai

Thank you for your interest in contributing to TUI Numerai! This document provides guidelines for contributing to the project.

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/hapticPaper/tui-numerai.git
cd tui-numerai
```

2. Install in development mode with all dependencies:
```bash
pip install -e ".[dev,wandb,influx]"
```

3. Run tests to ensure everything works:
```bash
pytest tests/
```

## Code Style

We use:
- **Black** for code formatting
- **Ruff** for linting
- **Type hints** for all function signatures

Format your code before submitting:
```bash
black .
ruff check .
```

## Adding a New Pipeline

To add a new pipeline:

1. Create a new file in `tui_numerai/pipelines/`:
```python
# tui_numerai/pipelines/my_pipeline.py

from ..core import Pipeline, PipelineConfig
from ..core.config import CompetitionType
from ..core.pipeline import register_pipeline

@register_pipeline("my_pipeline")
class MyPipeline(Pipeline):
    @classmethod
    def get_default_config(cls) -> PipelineConfig:
        return PipelineConfig(
            name="my_pipeline",
            competition=CompetitionType.NUMERAI,
            description="My custom pipeline",
        )
    
    def load_data(self):
        # Implement data loading
        pass
    
    def train(self):
        # Implement training logic
        pass
    
    def predict(self, data):
        # Implement prediction logic
        pass
```

2. Add import to `tui_numerai/pipelines/__init__.py`:
```python
from .my_pipeline import MyPipeline

__all__ = [..., "MyPipeline"]
```

3. Add import to `tui_numerai/cli.py`:
```python
from .pipelines import (
    ...,
    MyPipeline,
)
```

4. Add tests in `tests/test_pipeline.py`

## Adding New TUI Components

To add a new widget:

1. Add to `tui_numerai/tui/widgets.py`:
```python
class MyWidget(Static):
    """Description of the widget."""
    
    def compose(self) -> ComposeResult:
        # Define widget layout
        pass
```

2. Export from `tui_numerai/tui/__init__.py`

3. Use in screens or app as needed

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_pipeline.py

# Run with coverage
pytest tests/ --cov=tui_numerai
```

### Writing Tests

Place tests in `tests/` directory. Follow existing patterns:

```python
def test_my_feature():
    """Test description."""
    # Arrange
    config = MyConfig(...)
    
    # Act
    result = my_function(config)
    
    # Assert
    assert result == expected
```

## Documentation

When adding features:

1. Update docstrings
2. Update relevant .md files (README, USAGE, ARCHITECTURE)
3. Add examples to `examples/` directory

## Pull Request Process

1. Create a feature branch:
```bash
git checkout -b feature/my-feature
```

2. Make your changes and commit:
```bash
git add .
git commit -m "Add my feature"
```

3. Run tests and ensure they pass:
```bash
pytest tests/
black .
ruff check .
```

4. Push and create a PR:
```bash
git push origin feature/my-feature
```

5. In the PR description, include:
   - What the change does
   - Why it's needed
   - Any breaking changes
   - Testing done

## Code Review Guidelines

Reviewers should check:

- [ ] Code follows project style
- [ ] Tests are included and pass
- [ ] Documentation is updated
- [ ] No security vulnerabilities
- [ ] Backward compatibility maintained
- [ ] Performance impact considered

## Questions?

Open an issue on GitHub or reach out to the maintainers.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
