"""Model training module outline with boilerplate functions."""

from dataclasses import dataclass

from .config import PipelineConfig


def load_feature_table(config):
    """Load the feature table from disk.

    Inputs:
        config: PipelineConfig containing feature_store_dir.
    Outputs:
        features_df: tabular dataset with target and feature columns.
    """
    raise NotImplementedError


def split_train_val(features_df, time_column="encounter_start", val_fraction=0.2):
    """Split features into train/validation sets.

    Inputs:
        features_df: DataFrame from load_feature_table.
        time_column: column name for chronological splitting.
        val_fraction: fraction of newest rows used for validation.
    Outputs:
        train_df, val_df: split datasets.
    """
    raise NotImplementedError


def train_baseline_model(train_df, target_column="encounter_duration_minutes"):
    """Fit a baseline model.

    Inputs:
        train_df: training dataset.
        target_column: target variable name.
    Outputs:
        model: trained model instance.
    """
    raise NotImplementedError


def evaluate_model(model, val_df, target_column="encounter_duration_minutes"):
    """Evaluate the model on validation data.

    Inputs:
        model: trained model instance.
        val_df: validation dataset.
        target_column: target variable name.
    Outputs:
        metrics: dict-like structure with evaluation results.
    """
    raise NotImplementedError


def save_artifacts(config, model, metrics):
    """Persist model artifacts and metrics.

    Inputs:
        config: PipelineConfig containing model_dir.
        model: trained model instance.
        metrics: evaluation results.
    Outputs:
        artifact_paths: dict of output locations.
    """
    raise NotImplementedError


@dataclass
class DefaultModelTrainer:
    """Boilerplate trainer orchestrating the functions above."""

    config: PipelineConfig

    def train_model(self):
        """Orchestrate training pipeline and write artifacts."""
        raise NotImplementedError
