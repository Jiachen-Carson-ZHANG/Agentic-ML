"""Customer-level splitting helpers for labeled uplift rows."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.uplift import UpliftProjectContract
from src.uplift.validation import determine_stratification


@dataclass(frozen=True)
class UpliftSplitFrames:
    """Concrete train/validation/test frames plus split diagnostics."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    strategy: str
    warnings: List[str] = field(default_factory=list)


def write_split_artifacts(
    split: UpliftSplitFrames,
    *,
    output_dir: str | Path,
    prefix: str = "uplift_split",
    file_format: str = "csv",
) -> dict[str, str]:
    """Persist split frames for reproducible inspection/debugging.

    The main experiment loop keeps splits in-memory, but phase-1 notebooks wrote the
    split frames to disk. This helper keeps that capability in the code pipeline
    without requiring notebooks.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    file_format = file_format.lower().strip()

    def _write(df: pd.DataFrame, name: str) -> str:
        path = output / f"{prefix}_{name}.{file_format}"
        if file_format == "parquet":
            # Parquet requires optional engines (pyarrow/fastparquet). Keep a clear
            # error surface rather than silently falling back.
            df.to_parquet(path, index=False)
        elif file_format == "csv":
            df.to_csv(path, index=False)
        else:
            raise ValueError(f"unsupported file_format: {file_format} (expected 'csv' or 'parquet')")
        return str(path)

    return {
        "train": _write(split.train, "train"),
        "validation": _write(split.validation, "validation"),
        "test": _write(split.test, "test"),
    }


def _safe_stratify(values: pd.Series | None, n_rows: int) -> pd.Series | None:
    if values is None or n_rows == 0:
        return None
    counts = values.value_counts()
    return values if not counts.empty and int(counts.min()) >= 2 else None


def split_labeled_uplift_frame(
    labeled_df: pd.DataFrame,
    contract: UpliftProjectContract,
) -> UpliftSplitFrames:
    """Split only labeled uplift_train rows; never reads scoring rows."""
    split_contract = contract.split_contract
    decision = determine_stratification(
        labeled_df,
        treatment_col=contract.treatment_column,
        target_col=contract.target_column,
        split_contract=split_contract,
    )

    indices = np.arange(len(labeled_df))
    val_test_fraction = split_contract.val_fraction + split_contract.test_fraction
    stratify = _safe_stratify(decision.key, len(labeled_df))

    if val_test_fraction <= 0:
        return UpliftSplitFrames(
            train=labeled_df.reset_index(drop=True),
            validation=labeled_df.iloc[[]].copy(),
            test=labeled_df.iloc[[]].copy(),
            strategy=decision.strategy,
            warnings=decision.warnings,
        )

    train_idx, rest_idx = train_test_split(
        indices,
        test_size=val_test_fraction,
        random_state=split_contract.random_seed,
        stratify=stratify,
    )

    if split_contract.test_fraction <= 0:
        val_idx = rest_idx
        test_idx = np.array([], dtype=int)
    elif split_contract.val_fraction <= 0:
        val_idx = np.array([], dtype=int)
        test_idx = rest_idx
    else:
        relative_test = split_contract.test_fraction / val_test_fraction
        rest_stratify = (
            _safe_stratify(decision.key.iloc[rest_idx], len(rest_idx))
            if decision.key is not None
            else None
        )
        val_idx, test_idx = train_test_split(
            rest_idx,
            test_size=relative_test,
            random_state=split_contract.random_seed,
            stratify=rest_stratify,
        )

    return UpliftSplitFrames(
        train=labeled_df.iloc[train_idx].reset_index(drop=True),
        validation=labeled_df.iloc[val_idx].reset_index(drop=True),
        test=labeled_df.iloc[test_idx].reset_index(drop=True),
        strategy=decision.strategy,
        warnings=decision.warnings,
    )


def diagnose_uplift_split(
    split: UpliftSplitFrames,
    contract: UpliftProjectContract,
    *,
    min_eval_rows: int = 100,
    max_rate_delta: float = 0.03,
) -> dict[str, Any]:
    """Return treatment/outcome balance diagnostics for one concrete split.

    These diagnostics are intentionally pre-model: they answer whether the
    validation/test partitions are usable evaluation surfaces before any learner
    has a chance to overfit or produce suspicious Qini values.
    """
    partitions = {
        "train": split.train,
        "validation": split.validation,
        "test": split.test,
    }
    full = pd.concat(
        [frame for frame in partitions.values() if not frame.empty],
        ignore_index=True,
    )
    reference = _partition_stats(
        full,
        treatment_col=contract.treatment_column,
        target_col=contract.target_column,
    )
    partition_stats = {
        name: _partition_stats(
            frame,
            treatment_col=contract.treatment_column,
            target_col=contract.target_column,
        )
        for name, frame in partitions.items()
    }

    warnings = list(split.warnings)
    for name in ["validation", "test"]:
        stats = partition_stats[name]
        if stats["n_rows"] == 0:
            continue
        if stats["n_rows"] < min_eval_rows:
            warnings.append(
                f"{name} has {stats['n_rows']} rows; minimum reliable evaluation rows is {min_eval_rows}"
            )
        for metric in ["treatment_rate", "target_rate"]:
            delta = abs((stats[metric] or 0.0) - (reference[metric] or 0.0))
            if delta > max_rate_delta:
                warnings.append(
                    f"{name} {metric} differs from full data by {delta:.4f}"
                )
        if stats["min_joint_count"] < 1:
            warnings.append(f"{name} is missing at least one treatment/outcome cell")

    return {
        "reliable": not warnings,
        "strategy": split.strategy,
        "warnings": warnings,
        "reference": reference,
        "partitions": partition_stats,
    }


def _partition_stats(
    frame: pd.DataFrame,
    *,
    treatment_col: str,
    target_col: str,
) -> dict[str, Any]:
    if frame.empty:
        return {
            "n_rows": 0,
            "treatment_rate": None,
            "target_rate": None,
            "treated_response_rate": None,
            "control_response_rate": None,
            "joint_counts": {},
            "min_joint_count": 0,
        }
    treatment = frame[treatment_col].astype(int)
    target = frame[target_col].astype(int)
    treated = target[treatment == 1]
    control = target[treatment == 0]
    joint_counts = {
        f"t{int(t_value)}_y{int(y_value)}": int(count)
        for (t_value, y_value), count in frame.groupby([treatment_col, target_col]).size().items()
    }
    all_joint_counts = [
        joint_counts.get(f"t{t_value}_y{y_value}", 0)
        for t_value in [0, 1]
        for y_value in [0, 1]
    ]
    return {
        "n_rows": int(len(frame)),
        "treatment_rate": round(float(treatment.mean()), 6),
        "target_rate": round(float(target.mean()), 6),
        "treated_response_rate": None
        if treated.empty
        else round(float(treated.mean()), 6),
        "control_response_rate": None
        if control.empty
        else round(float(control.mean()), 6),
        "joint_counts": joint_counts,
        "min_joint_count": int(min(all_joint_counts)),
    }
