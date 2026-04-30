"""Deterministic pre-run tuning helpers for uplift trials."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from src.models.uplift import UpliftExperimentRecord, UpliftTrialSpec
from src.uplift.metrics import normalized_qini_auc_score


_REGULARIZED_PARAM_SETS: dict[str, list[dict[str, object]]] = {
    "logistic_regression": [
        {"C": 0.3, "max_iter": 1000},
        {"C": 1.0, "max_iter": 1000},
        {"C": 3.0, "max_iter": 1000},
    ],
    "gradient_boosting": [
        {
            "n_estimators": 200,
            "learning_rate": 0.03,
            "max_depth": 2,
            "min_samples_leaf": 50,
            "subsample": 0.7,
        },
        {
            "n_estimators": 120,
            "learning_rate": 0.03,
            "max_depth": 2,
            "min_samples_leaf": 100,
            "subsample": 0.7,
        },
    ],
    "random_forest": [
        {
            "n_estimators": 200,
            "max_depth": 6,
            "min_samples_leaf": 50,
            "max_features": "sqrt",
            "n_jobs": -1,
        },
        {
            "n_estimators": 300,
            "max_depth": 4,
            "min_samples_leaf": 100,
            "max_features": "sqrt",
            "n_jobs": -1,
        },
    ],
    "xgboost": [
        {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 10,
            "reg_lambda": 10.0,
        },
        {
            "n_estimators": 400,
            "max_depth": 2,
            "learning_rate": 0.03,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "min_child_weight": 20,
            "reg_lambda": 10.0,
        },
    ],
    "lightgbm": [
        {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 10.0,
        },
        {
            "n_estimators": 400,
            "max_depth": 3,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "min_child_samples": 100,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_lambda": 10.0,
        },
    ],
}


def build_pre_run_tuning_specs(
    base_spec: UpliftTrialSpec,
    *,
    split_seeds: Sequence[int] = (42, 7, 99, 123),
    max_param_sets: int = 2,
) -> list[UpliftTrialSpec]:
    """Expand one planned trial into deterministic param/seed candidates."""
    param_sets = _REGULARIZED_PARAM_SETS.get(base_spec.base_estimator, [base_spec.params])
    param_sets = param_sets[: max(1, max_param_sets)]
    specs: list[UpliftTrialSpec] = []
    for param_index, params in enumerate(param_sets, start=1):
        for seed in _unique_seeds(split_seeds, base_spec.split_seed):
            specs.append(
                base_spec.model_copy(
                    update={
                        "spec_id": f"{base_spec.spec_id}__tune_p{param_index}_s{seed}",
                        "hypothesis_id": f"{base_spec.hypothesis_id}__tune_p{param_index}_s{seed}",
                        "params": dict(params),
                        "split_seed": int(seed),
                    }
                )
            )
    return specs


def select_stable_tuning_record(
    records: Iterable[UpliftExperimentRecord],
) -> UpliftExperimentRecord | None:
    """Choose the candidate with the best stability-adjusted normalized Qini."""
    successful = [record for record in records if record.status == "success"]
    if not successful:
        return None
    return max(successful, key=_stable_record_score)


def tuning_summary(records: Iterable[UpliftExperimentRecord]) -> list[dict[str, object]]:
    """Return a compact summary for audit artifacts and logs."""
    rows: list[dict[str, object]] = []
    for record in records:
        val = _normalized_qini_from_record(record, "uplift_scores")
        held = _normalized_qini_from_record(record, "held_out_predictions")
        rows.append(
            {
                "run_id": record.run_id,
                "hypothesis_id": record.hypothesis_id,
                "status": record.status,
                "learner_family": record.uplift_learner_family,
                "base_estimator": record.base_estimator,
                "split_seed": record.split_seed,
                "params_hash": record.params_hash,
                "val_normalized_qini": val,
                "held_out_normalized_qini": held,
                "stable_score": _stable_record_score(record),
                "error": record.error,
            }
        )
    return rows


def write_tuning_summary(
    path: str | Path,
    records: Iterable[UpliftExperimentRecord],
) -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(tuning_summary(records), indent=2),
        encoding="utf-8",
    )
    return str(output_path)


def _stable_record_score(record: UpliftExperimentRecord) -> float:
    val = _normalized_qini_from_record(record, "uplift_scores")
    held = _normalized_qini_from_record(record, "held_out_predictions")
    if val is not None and held is not None:
        return min(val, held) - 0.25 * abs(held - val)
    if val is not None:
        return val
    if record.qini_auc is not None:
        return float(record.qini_auc)
    return float("-inf")


def _normalized_qini_from_record(
    record: UpliftExperimentRecord,
    artifact_key: str,
) -> float | None:
    path = record.artifact_paths.get(artifact_key)
    if not path:
        return None
    try:
        scores = pd.read_csv(path)
        return normalized_qini_auc_score(
            scores["target"].to_numpy(),
            scores["treatment_flg"].to_numpy(),
            scores["uplift"].to_numpy(),
        )
    except Exception:
        return None


def _unique_seeds(split_seeds: Sequence[int], fallback_seed: int) -> list[int]:
    seeds = [int(seed) for seed in split_seeds] or [int(fallback_seed)]
    if int(fallback_seed) not in seeds:
        seeds.insert(0, int(fallback_seed))
    return list(dict.fromkeys(seeds))
