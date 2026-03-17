from __future__ import annotations
import os
import time
import pandas as pd
from src.models.task import RunConfig
from src.models.results import RunResult
from src.execution.result_parser import ResultParser


class AutoGluonRunner:
    """Runs AutoGluon TabularPredictor for a given RunConfig."""

    def __init__(self, target_column: str) -> None:
        self.target_column = target_column

    def run(self, config: RunConfig) -> RunResult:
        try:
            from autogluon.tabular import TabularPredictor
        except ImportError:
            return ResultParser.from_error(
                run_id=config.run_id,
                error_msg="AutoGluon not installed. Run: pip install autogluon.tabular",
                artifacts_dir=config.output_dir,
            )

        os.makedirs(config.output_dir, exist_ok=True)
        df = pd.read_csv(config.data_path)

        kwargs = dict(config.autogluon_kwargs)
        kwargs["path"] = config.output_dir

        predictor = TabularPredictor(
            label=self.target_column,
            **{k: v for k, v in kwargs.items()
               if k in ("eval_metric", "path", "problem_type")}
        )

        fit_kwargs = {k: v for k, v in kwargs.items()
                      if k not in ("eval_metric", "path", "problem_type")}

        start = time.time()
        try:
            predictor.fit(df, **fit_kwargs)
        except Exception as e:
            return ResultParser.from_error(
                run_id=config.run_id,
                error_msg=str(e),
                artifacts_dir=config.output_dir,
            )
        fit_time = time.time() - start

        # Get validation score from leaderboard
        lb = predictor.leaderboard(silent=True)
        primary_metric = float(lb["score_val"].max()) if not lb.empty else 0.0

        return ResultParser.from_predictor(
            predictor=predictor,
            run_id=config.run_id,
            fit_time=fit_time,
            artifacts_dir=config.output_dir,
            primary_metric_value=primary_metric,
        )
