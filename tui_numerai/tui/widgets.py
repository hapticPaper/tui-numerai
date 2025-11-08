"""Reusable TUI widgets."""

from typing import Any, Dict, List, Optional

from rich.syntax import Syntax
from rich.table import Table
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.reactive import reactive
from textual.widgets import Static, RichLog, Label, Input, Button, DataTable


class OutputCapture(Static):
    """Widget for capturing and displaying scrolling output.
    
    Useful for capturing logs from libraries like LightGBM.
    """

    def __init__(self, max_lines: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.max_lines = max_lines
        self._log_widget: Optional[RichLog] = None

    def compose(self) -> ComposeResult:
        self._log_widget = RichLog(highlight=True, markup=True)
        self._log_widget.max_lines = self.max_lines
        yield self._log_widget

    def write(self, text: str) -> None:
        """Write text to the output capture."""
        if self._log_widget:
            self._log_widget.write(text)

    def clear(self) -> None:
        """Clear all captured output."""
        if self._log_widget:
            self._log_widget.clear()


class MetricsDisplay(Static):
    """Widget for displaying training metrics in a table."""

    metrics: reactive[Dict[str, Any]] = reactive({})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._table_widget: Optional[DataTable] = None

    def compose(self) -> ComposeResult:
        self._table_widget = DataTable()
        self._table_widget.add_columns("Metric", "Value")
        yield self._table_widget

    def watch_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update the metrics display when metrics change."""
        if not self._table_widget:
            return

        self._table_widget.clear()
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted_value = f"{value:.6f}"
            else:
                formatted_value = str(value)
            self._table_widget.add_row(key, formatted_value)

    def update_metrics(self, new_metrics: Dict[str, Any]) -> None:
        """Update metrics with new values."""
        self.metrics = {**self.metrics, **new_metrics}


class ParameterEditor(Vertical):
    """Widget for editing pipeline parameters."""

    def __init__(self, parameters: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.parameters = parameters
        self._inputs: Dict[str, Input] = {}

    def compose(self) -> ComposeResult:
        yield Label("Pipeline Parameters", classes="section-header")
        
        for key, value in self.parameters.items():
            with Horizontal(classes="parameter-row"):
                yield Label(f"{key}:", classes="parameter-label")
                input_widget = Input(
                    value=str(value),
                    placeholder=f"Enter {key}",
                    id=f"param_{key}",
                )
                self._inputs[key] = input_widget
                yield input_widget
        
        with Horizontal(classes="button-row"):
            yield Button("Apply", variant="primary", id="apply_params")
            yield Button("Reset", variant="default", id="reset_params")

    def get_parameters(self) -> Dict[str, Any]:
        """Get the current parameter values.
        
        Returns:
            Dictionary of parameter values with type conversion
        """
        result = {}
        for key, input_widget in self._inputs.items():
            value = input_widget.value
            original_type = type(self.parameters[key])
            
            # Try to convert to the original type
            try:
                if original_type == bool:
                    result[key] = value.lower() in ("true", "1", "yes", "y")
                elif original_type in (int, float):
                    result[key] = original_type(value)
                else:
                    result[key] = value
            except (ValueError, AttributeError):
                result[key] = value
        
        return result

    def reset_parameters(self) -> None:
        """Reset all parameters to their original values."""
        for key, input_widget in self._inputs.items():
            input_widget.value = str(self.parameters[key])


class RunSelector(Vertical):
    """Widget for selecting previous runs to resume."""

    def __init__(self, runs: List[Dict[str, Any]], **kwargs):
        super().__init__(**kwargs)
        self.runs = runs
        self._table_widget: Optional[DataTable] = None

    def compose(self) -> ComposeResult:
        yield Label("Previous Runs", classes="section-header")
        
        self._table_widget = DataTable()
        self._table_widget.add_columns("Run ID", "Status", "Created", "Completed")
        self._table_widget.cursor_type = "row"
        
        for run in self.runs:
            self._table_widget.add_row(
                run["run_id"],
                run["status"],
                run["created_at"][:19] if run.get("created_at") else "N/A",
                run["completed_at"][:19] if run.get("completed_at") else "N/A",
            )
        
        yield self._table_widget
        
        with Horizontal(classes="button-row"):
            yield Button("Resume Selected", variant="primary", id="resume_run")
            yield Button("New Run", variant="default", id="new_run")

    def get_selected_run_id(self) -> Optional[str]:
        """Get the run ID of the selected row.
        
        Returns:
            Selected run ID or None
        """
        if not self._table_widget or self._table_widget.cursor_row < 0:
            return None
        
        row = self._table_widget.cursor_row
        if row < len(self.runs):
            return self.runs[row]["run_id"]
        return None


class PipelineSelector(Vertical):
    """Widget for selecting a pipeline and competition."""

    def __init__(self, pipelines: List[Dict[str, Any]], **kwargs):
        super().__init__(**kwargs)
        self.pipelines = pipelines
        self._table_widget: Optional[DataTable] = None

    def compose(self) -> ComposeResult:
        yield Label("Available Pipelines", classes="section-header")
        
        self._table_widget = DataTable()
        self._table_widget.add_columns("Pipeline", "Competition", "Version", "Description")
        self._table_widget.cursor_type = "row"
        
        for pipeline in self.pipelines:
            self._table_widget.add_row(
                pipeline["name"],
                pipeline["competition"],
                pipeline["version"],
                pipeline["description"][:40] + "..." if len(pipeline["description"]) > 40 else pipeline["description"],
            )
        
        yield self._table_widget
        
        with Horizontal(classes="button-row"):
            yield Button("Select", variant="primary", id="select_pipeline")
            yield Button("Quit", variant="error", id="quit_app")

    def get_selected_pipeline(self) -> Optional[str]:
        """Get the name of the selected pipeline.
        
        Returns:
            Selected pipeline name or None
        """
        if not self._table_widget or self._table_widget.cursor_row < 0:
            return None
        
        row = self._table_widget.cursor_row
        if row < len(self.pipelines):
            return self.pipelines[row]["name"]
        return None
