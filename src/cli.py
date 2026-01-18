"""CLI entry points for orchestrating the pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline_architecture import build_pipeline


def main() -> None:
    parser = argparse.ArgumentParser("ed-pipeline")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=["ingest", "features", "train", "evaluate", "all"], default="all")
    args = parser.parse_args()

    pipeline = build_pipeline(args.project_root)

    if args.mode == "ingest":
        pipeline.ingestion.load_raw_data()
        pipeline.ingestion.clean_data()
    elif args.mode == "features":
        pipeline.feature_pipeline.build_features()
    elif args.mode == "train":
        pipeline.trainer.train_model()
    elif args.mode == "evaluate":
        pipeline.evaluator.run_evaluation()
    else:
        pipeline.run()


if __name__ == "__main__":
    main()
