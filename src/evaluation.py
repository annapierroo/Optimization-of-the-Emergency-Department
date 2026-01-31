"""Model evaluation components."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .config import PipelineConfig


class EvaluatorPort(Protocol):
    """Computes metrics and diagnostics."""

    def run_evaluation(self) -> None:
        """Store metric reports."""


@dataclass
class DefaultEvaluator(EvaluatorPort):
    """Placeholder evaluator."""

    config: PipelineConfig

    def run_evaluation(self) -> None:
        pass

