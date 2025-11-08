"""PyTorch neural network pipeline for Numerai."""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..core import Pipeline, PipelineConfig, RunConfig
from ..core.config import CompetitionType
from ..core.pipeline import register_pipeline


class NumeraiNet(nn.Module):
    """Simple neural network for Numerai."""

    def __init__(self, input_dim: int, hidden_dims: list = [256, 128, 64]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()


@register_pipeline("pytorch_numerai")
class PyTorchPipeline(Pipeline):
    """PyTorch neural network pipeline for Numerai competition."""

    def __init__(self, config: PipelineConfig, run_config: RunConfig):
        super().__init__(config, run_config)
        self.feature_cols = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def get_default_config(cls) -> PipelineConfig:
        """Get default configuration for PyTorch pipeline."""
        return PipelineConfig(
            name="pytorch_numerai",
            competition=CompetitionType.NUMERAI,
            version="1.0.0",
            description="PyTorch neural network for Numerai Classic",
            epochs=50,
            batch_size=1024,
            learning_rate=0.001,
            model_params={
                "hidden_dims": [256, 128, 64],
            },
            target_name="target",
            feature_set="all",
        )

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and validation data."""
        self.emit_event("log", {"message": "Loading data..."})

        # Create dummy data for demonstration
        n_samples = 10000
        n_features = 20

        self.feature_cols = [f"feature_{i}" for i in range(n_features)]

        # Generate random data
        np.random.seed(42)
        train_features = np.random.randn(n_samples, n_features).astype(np.float32)
        train_target = np.random.rand(n_samples).astype(np.float32)

        train_data = pd.DataFrame(train_features, columns=self.feature_cols)
        train_data["era"] = np.repeat(np.arange(n_samples // 100), 100)
        train_data[self.config.target_name] = train_target

        val_size = n_samples // 4
        val_features = np.random.randn(val_size, n_features).astype(np.float32)
        val_target = np.random.rand(val_size).astype(np.float32)

        val_data = pd.DataFrame(val_features, columns=self.feature_cols)
        n_val_eras = max(1, val_size // 100)
        val_data["era"] = np.repeat(np.arange(n_val_eras), val_size // n_val_eras)[:val_size]
        val_data[self.config.target_name] = val_target

        self.emit_event(
            "log",
            {
                "message": f"Loaded {len(train_data)} training samples, {len(val_data)} validation samples"
            },
        )

        return train_data, val_data

    def train(self) -> Dict[str, Any]:
        """Train the PyTorch model."""
        self.emit_event("log", {"message": f"Starting PyTorch training on {self.device}..."})

        # Load data
        train_data, val_data = self.load_data()
        self.training_data = train_data
        self.validation_data = val_data

        # Prepare data
        X_train = torch.FloatTensor(train_data[self.feature_cols].values)
        y_train = torch.FloatTensor(train_data[self.config.target_name].values)
        X_val = torch.FloatTensor(val_data[self.feature_cols].values)
        y_val = torch.FloatTensor(val_data[self.config.target_name].values)

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        # Create model
        input_dim = len(self.feature_cols)
        self.model = NumeraiNet(
            input_dim,
            hidden_dims=self.config.model_params.get("hidden_dims", [256, 128, 64]),
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        # Training loop
        best_val_loss = float("inf")

        for epoch in range(self.config.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation phase
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            # Log progress
            if (epoch + 1) % 10 == 0:
                self.emit_event(
                    "log",
                    {
                        "message": f"Epoch {epoch+1}/{self.config.epochs} - "
                        f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                    },
                )

                self.emit_event(
                    "metrics",
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    },
                )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if self.config.save_model:
                    checkpoint_path = self.run_config.run_dir / "checkpoints" / "best_model.pt"
                    torch.save(self.model.state_dict(), checkpoint_path)

        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            train_preds = self.model(X_train.to(self.device)).cpu().numpy()
            val_preds = self.model(X_val.to(self.device)).cpu().numpy()

        # Calculate correlations
        train_corr = np.corrcoef(y_train.numpy(), train_preds)[0, 1]
        val_corr = np.corrcoef(y_val.numpy(), val_preds)[0, 1]

        metrics = {
            "train_correlation": float(train_corr),
            "val_correlation": float(val_corr),
            "best_val_loss": float(best_val_loss),
            "final_train_loss": float(train_loss),
        }

        self.emit_event("metrics", metrics)
        self.save_metrics(metrics)

        # Save final model
        if self.config.save_model:
            model_path = self.run_config.run_dir / "model.pt"
            torch.save(self.model.state_dict(), model_path)
            self.emit_event("log", {"message": f"Model saved to {model_path}"})

        # Save predictions
        if self.config.save_predictions:
            val_data["prediction"] = val_preds
            pred_path = self.save_predictions(val_data[["prediction"]])
            self.emit_event("log", {"message": f"Predictions saved to {pred_path}"})

        self.emit_event("log", {"message": "Training complete!"})

        return metrics

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for new data."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        self.model.eval()
        X = torch.FloatTensor(data[self.feature_cols].values).to(self.device)

        with torch.no_grad():
            predictions = self.model(X).cpu().numpy()

        result = pd.DataFrame({"prediction": predictions}, index=data.index)
        return result

    def save_model(self, path: Path = None) -> Path:
        """Save the trained model."""
        if path is None:
            path = self.run_config.run_dir / "model.pt"

        torch.save(self.model.state_dict(), path)
        self.logger.info("model_saved", path=str(path))
        return path

    def load_model(self, path: Path) -> None:
        """Load a trained model from disk."""
        # Need to know model architecture
        if self.feature_cols is None:
            raise ValueError("Feature columns not set. Load data first.")

        input_dim = len(self.feature_cols)
        self.model = NumeraiNet(
            input_dim,
            hidden_dims=self.config.model_params.get("hidden_dims", [256, 128, 64]),
        ).to(self.device)

        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.logger.info("model_loaded", path=str(path))
