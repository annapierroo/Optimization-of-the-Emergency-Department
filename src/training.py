"""Model training components."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .config import PipelineConfig


class ModelTrainerPort(Protocol):
    """Runs experiment training loops."""

    def train_model(self) -> None:
        """Produce serialized model artifacts."""


@dataclass
class DefaultModelTrainer(ModelTrainerPort):
    """Placeholder trainer."""

    config: PipelineConfig

    def train_model(self) -> None:
        pass

