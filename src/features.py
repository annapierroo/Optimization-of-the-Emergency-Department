"\"\"Feature engineering interfaces.\"\""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .config import PipelineConfig


class FeaturePipelinePort(Protocol):
    """Creates model-ready feature sets."""

    def build_features(self) -> None:
        """Transform cleaned logs to feature tables."""


@dataclass
class DefaultFeaturePipeline(FeaturePipelinePort):
    """Placeholder implementation."""

    config: PipelineConfig

    def build_features(self) -> None:
        pass

