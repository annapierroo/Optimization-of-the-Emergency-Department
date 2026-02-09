"""Pipeline orchestration for ingestion, feature building, and model training."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from . import ingest_data
from .config import PipelineConfig, default_config
from .evaluation import DefaultEvaluator, EvaluatorPort
from .features import DefaultFeaturePipeline, FeaturePipelinePort


class IngestionPort(Protocol):
    """Defines how raw hospital data enters the system."""

    def load_raw_data(self) -> None:
        """Pull files from storage."""

    def clean_data(self) -> None:
        """Normalize and persist cleaned encounter logs."""


class ModelTrainerPort(Protocol):
    """Interface for any train_* module adapter."""

    name: str

    def train_model(self) -> None:
        """Run model training for a single task."""


@dataclass
class DefaultIngestion(IngestionPort):
    """Delegates to the existing ingest_data module."""

    config: PipelineConfig

    def load_raw_data(self) -> None:
        pass

    def clean_data(self) -> None:
        ingest_data.ingest_and_clean()


@dataclass
class WaitTimeTrainerAdapter(ModelTrainerPort):
    """Adapter for src.train_wait_time.DefaultModelTrainer."""

    config: PipelineConfig
    name: str = "wait_time"

    def train_model(self) -> None:
        # Lazy import keeps orchestration importable without heavy ML deps at import time.
        from .train_wait_time import DefaultModelTrainer

        DefaultModelTrainer(config=self.config).train_model()


@dataclass
class LosTrainerAdapter(ModelTrainerPort):
    """Adapter for src.train_los_models.train_los_models."""

    config: PipelineConfig
    name: str = "los"

    def train_model(self) -> None:
        from .train_los_models import train_los_models

        train_los_models(config=self.config)


@dataclass
class NextActivityTrainerAdapter(ModelTrainerPort):
    """Adapter for src.train_next_activity.train_next_activity."""

    config: PipelineConfig
    name: str = "next_activity"

    def train_model(self) -> None:
        from .train_next_activity import train_next_activity

        train_next_activity(config=self.config)


@dataclass
class EmergencyDepartmentPipeline:
    """Orchestrates each stage in sequence."""

    config: PipelineConfig
    ingestion: IngestionPort
    feature_pipeline: FeaturePipelinePort
    trainers: list[ModelTrainerPort]
    evaluator: EvaluatorPort

    def run(self) -> None:
        self.ingestion.load_raw_data()
        self.ingestion.clean_data()
        self.feature_pipeline.build_features()
        for trainer in self.trainers:
            trainer.train_model()
        self.evaluator.run_evaluation()


def build_trainer_registry(config: PipelineConfig) -> dict[str, ModelTrainerPort]:
    """Return available trainer adapters."""

    return {
        "wait_time": WaitTimeTrainerAdapter(config=config),
        "los": LosTrainerAdapter(config=config),
        "next_activity": NextActivityTrainerAdapter(config=config),
    }


def build_pipeline(data_root: Path, trainer_names: list[str] | None = None) -> EmergencyDepartmentPipeline:
    """Factory producing pipeline with default components."""

    config = default_config(data_root)
    registry = build_trainer_registry(config)
    if trainer_names is None:
        trainer_names = list(registry.keys())

    return EmergencyDepartmentPipeline(
        config=config,
        ingestion=DefaultIngestion(config),
        feature_pipeline=DefaultFeaturePipeline(config),
        trainers=[registry[name] for name in trainer_names],
        evaluator=DefaultEvaluator(config),
    )
