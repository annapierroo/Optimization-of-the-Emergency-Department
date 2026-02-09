"""Single entrypoint to train project models."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src import ingest_data
from src.config import default_config
from src.features import DefaultFeaturePipeline
from src.pipeline_architecture import build_trainer_registry

LOGGER = logging.getLogger(__name__)


def run_training(
    project_root: Path,
    trainer_names: list[str] | None = None,
    run_ingestion: bool = True,
    run_features: bool = True,
) -> None:
    """Run data prep and selected training pipelines."""
    config = default_config(project_root)

    if run_ingestion:
        LOGGER.info("Running ingestion")
        ingest_data.ingest_and_clean()

    if run_features:
        LOGGER.info("Building features")
        DefaultFeaturePipeline(config).build_features()

    registry = build_trainer_registry(config)
    if trainer_names is None:
        trainer_names = ["wait_time", "los", "next_activity"]

    for name in trainer_names:
        if name not in registry:
            raise ValueError(f"Unknown trainer '{name}'. Available: {sorted(registry)}")
        LOGGER.info("Training model: %s", name)
        registry[name].train_model()

    LOGGER.info("Training run completed for: %s", ", ".join(trainer_names))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("run_training")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root containing data/, src/, artifacts/",
    )
    parser.add_argument(
        "--trainers",
        nargs="+",
        default=["wait_time", "los", "next_activity"],
        help="Subset of trainers to run (wait_time los next_activity)",
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip ingestion step",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip feature build step",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = _parse_args()
    run_training(
        project_root=args.project_root,
        trainer_names=args.trainers,
        run_ingestion=not args.skip_ingestion,
        run_features=not args.skip_features,
    )
