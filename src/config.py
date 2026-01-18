"""Configuration objects and utilities for the emergency-department project."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Root directories and resource identifiers."""

    project_root: Path
    raw_data_dir: Path
    processed_data_dir: Path
    feature_store_dir: Path
    model_dir: Path
    reports_dir: Path


def default_config(project_root: Path) -> PipelineConfig:
    """Return default directory layout."""

    return PipelineConfig(
        project_root=project_root,
        raw_data_dir=project_root / "data" / "raw",
        processed_data_dir=project_root / "data" / "processed",
        feature_store_dir=project_root / "data" / "features",
        model_dir=project_root / "artifacts" / "models",
        reports_dir=project_root / "reports",
    )

