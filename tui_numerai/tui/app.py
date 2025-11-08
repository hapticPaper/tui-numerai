"""Main TUI application."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Label, Static

from ..core import PipelineRegistry, RunConfig, RunStateManager
from .widgets import (
    MetricsDisplay,
    OutputCapture,
    ParameterEditor,
    PipelineSelector,
    RunSelector,
)


logger = structlog.get_logger()


class PipelineSelectionScreen(Screen):
    """Screen for selecting a pipeline."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Label("Welcome to TUI Numerai", id="title"),
            PipelineSelector(
                self.app.get_pipeline_list(),
                id="pipeline_selector",
            ),
            id="main_container",
        )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "select_pipeline":
            selector = self.query_one("#pipeline_selector", PipelineSelector)
            pipeline_name = selector.get_selected_pipeline()
            if pipeline_name:
                self.app.selected_pipeline = pipeline_name
                self.app.push_screen("run_selection")
        elif event.button.id == "quit_app":
            self.app.exit()


class RunSelectionScreen(Screen):
    """Screen for selecting or creating a run."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        
        pipeline_name = self.app.selected_pipeline
        runs = self.app.state_manager.get_resumable_runs(pipeline_name)
        
        yield Container(
            Label(f"Pipeline: {pipeline_name}", id="title"),
            RunSelector(runs, id="run_selector") if runs else Label("No previous runs available"),
            id="main_container",
        )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "resume_run":
            selector = self.query_one("#run_selector", RunSelector)
            run_id = selector.get_selected_run_id()
            if run_id:
                self.app.resume_run(run_id)
        elif event.button.id == "new_run":
            self.app.create_new_run()

    def action_back(self) -> None:
        """Go back to pipeline selection."""
        self.app.pop_screen()


class TrainingScreen(Screen):
    """Screen for monitoring training."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("s", "stop", "Stop Training"),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_task: Optional[asyncio.Task] = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Label(f"Training: {self.app.selected_pipeline}", id="training_title"),
            Horizontal(
                Vertical(
                    Label("Training Output", classes="section-header"),
                    OutputCapture(id="output_capture"),
                    id="output_section",
                ),
                Vertical(
                    Label("Metrics", classes="section-header"),
                    MetricsDisplay(id="metrics_display"),
                    id="metrics_section",
                ),
                id="training_container",
            ),
            Horizontal(
                Button("Stop Training", variant="error", id="stop_training"),
                Button("Back to Menu", variant="default", id="back_menu"),
                id="button_container",
            ),
            id="main_container",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Start training when screen mounts."""
        self.training_task = asyncio.create_task(self.run_training())

    async def run_training(self) -> None:
        """Run the training pipeline."""
        try:
            output = self.query_one("#output_capture", OutputCapture)
            metrics_display = self.query_one("#metrics_display", MetricsDisplay)
            
            output.write("[bold green]Starting training...[/bold green]")
            
            # Get pipeline
            pipeline_class = PipelineRegistry.get(self.app.selected_pipeline)
            pipeline = pipeline_class(
                self.app.current_pipeline_config,
                self.app.current_run_config,
            )
            
            # Add callback to update UI
            def on_event(event_type: str, data: Dict[str, Any]) -> None:
                if event_type == "log":
                    output.write(data.get("message", ""))
                elif event_type == "metrics":
                    metrics_display.update_metrics(data)
            
            pipeline.add_callback(on_event)
            
            # Run training in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, pipeline.train)
            
            output.write(f"[bold green]Training completed![/bold green]")
            metrics_display.update_metrics(results)
            
            # Update run config
            self.app.current_run_config.status = "completed"
            self.app.current_run_config.completed_at = datetime.now()
            self.app.state_manager.save_run_config(self.app.current_run_config)
            
        except Exception as e:
            output = self.query_one("#output_capture", OutputCapture)
            output.write(f"[bold red]Error: {str(e)}[/bold red]")
            logger.error("training_error", error=str(e), exc_info=True)
            
            self.app.current_run_config.status = "failed"
            self.app.state_manager.save_run_config(self.app.current_run_config)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "stop_training":
            if self.training_task:
                self.training_task.cancel()
            self.app.pop_screen()
        elif event.button.id == "back_menu":
            if self.training_task:
                self.training_task.cancel()
            self.app.pop_screen()

    def action_stop(self) -> None:
        """Stop training."""
        if self.training_task:
            self.training_task.cancel()
        self.app.pop_screen()


class NumeraiTUI(App):
    """Main TUI application for Numerai training."""

    CSS = """
    #title {
        text-align: center;
        padding: 1;
        background: $primary;
        color: $text;
    }
    
    #training_title {
        text-align: center;
        padding: 1;
        background: $accent;
        color: $text;
    }
    
    .section-header {
        text-style: bold;
        background: $surface;
        padding: 1;
    }
    
    #main_container {
        height: 100%;
        padding: 1;
    }
    
    #training_container {
        height: 1fr;
    }
    
    #output_section {
        width: 2fr;
        height: 100%;
        border: solid $primary;
        padding: 1;
    }
    
    #metrics_section {
        width: 1fr;
        height: 100%;
        border: solid $accent;
        padding: 1;
    }
    
    #button_container {
        height: auto;
        padding: 1;
        align: center middle;
    }
    
    .button-row {
        height: auto;
        align: center middle;
        padding: 1;
    }
    
    .parameter-row {
        height: auto;
        padding: 0 1;
    }
    
    .parameter-label {
        width: 30%;
        padding: 1;
    }
    
    OutputCapture {
        height: 100%;
    }
    
    MetricsDisplay {
        height: 100%;
    }
    
    DataTable {
        height: 1fr;
    }
    """

    SCREENS = {
        "pipeline_selection": PipelineSelectionScreen,
        "run_selection": RunSelectionScreen,
        "training": TrainingScreen,
    }

    def __init__(self, runs_dir: Path = Path("./runs"), **kwargs):
        super().__init__(**kwargs)
        self.state_manager = RunStateManager(runs_dir)
        self.selected_pipeline: Optional[str] = None
        self.current_run_config: Optional[RunConfig] = None
        self.current_pipeline_config = None

    def on_mount(self) -> None:
        """Initialize the application."""
        self.push_screen("pipeline_selection")

    def get_pipeline_list(self) -> list:
        """Get list of available pipelines."""
        pipelines = []
        for name in PipelineRegistry.list_pipelines():
            info = PipelineRegistry.get_pipeline_info(name)
            pipelines.append(info)
        return pipelines

    def create_new_run(self) -> None:
        """Create a new training run."""
        if not self.selected_pipeline:
            return

        # Get pipeline class and default config
        pipeline_class = PipelineRegistry.get(self.selected_pipeline)
        pipeline_config = pipeline_class.get_default_config()
        
        # Create run config
        run_id = self.state_manager.create_run_id(self.selected_pipeline)
        run_dir = self.state_manager.get_run_directory(run_id)
        
        run_config = RunConfig(
            run_id=run_id,
            pipeline_name=self.selected_pipeline,
            pipeline_version=pipeline_config.version,
            competition=pipeline_config.competition,
            pipeline_config=pipeline_config,
            run_dir=run_dir,
        )
        
        run_config.ensure_run_directory()
        self.state_manager.save_run_config(run_config)
        
        self.current_run_config = run_config
        self.current_pipeline_config = pipeline_config
        
        # Start training
        self.push_screen("training")

    def resume_run(self, run_id: str) -> None:
        """Resume a previous training run."""
        run_config = self.state_manager.load_run_config(run_id)
        run_config.status = "resumed"
        run_config.resume_from = run_id
        
        self.current_run_config = run_config
        self.current_pipeline_config = run_config.pipeline_config
        
        self.state_manager.save_run_config(run_config)
        
        # Start training
        self.push_screen("training")
