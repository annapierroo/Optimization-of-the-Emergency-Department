"""Architecture outline with boilerplate services and blank functions."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from . import ingest_data
from .config import PipelineConfig, default_config
from .evaluation import DefaultEvaluator, EvaluatorPort
from .features import DefaultFeaturePipeline, FeaturePipelinePort
from .training import DefaultModelTrainer, ModelTrainerPort


class IngestionPort(Protocol):
    """Defines how raw hospital data enters the system."""

    def load_raw_data(self) -> None:
        """Pull files from storage."""

    def clean_data(self) -> None:
        """Normalize and persist cleaned encounter logs."""


@dataclass
class DefaultIngestion(IngestionPort):
    """Delegates to the existing ingest_data module."""

    config: PipelineConfig

    def load_raw_data(self) -> None:
        pass

    def clean_data(self) -> None:
        ingest_data.ingest_and_clean()


@dataclass
class EmergencyDepartmentPipeline:
    """Orchestrates each stage in sequence."""

    config: PipelineConfig
    ingestion: IngestionPort
    feature_pipeline: FeaturePipelinePort
    trainer: ModelTrainerPort
    evaluator: EvaluatorPort

    def run(self) -> None:
        self.ingestion.load_raw_data()
        self.ingestion.clean_data()
        self.feature_pipeline.build_features()
        self.trainer.train_model()
        self.evaluator.run_evaluation()


def build_pipeline(data_root: Path) -> EmergencyDepartmentPipeline:
    """Factory producing pipeline with default components."""

    config = default_config(data_root)

    return EmergencyDepartmentPipeline(
        config=config,
        ingestion=DefaultIngestion(config),
        feature_pipeline=DefaultFeaturePipeline(config),
        trainer=DefaultModelTrainer(config),
        evaluator=DefaultEvaluator(config),
    )
