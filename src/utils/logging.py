"""Logging helpers."""
from __future__ import annotations

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Basic logging configuration."""

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=level,
    )

