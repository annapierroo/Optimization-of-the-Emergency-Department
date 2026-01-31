"""File-system helper functions."""
from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> None:
    """Create directory if missing."""

    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    """Write small text payloads."""

    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")

