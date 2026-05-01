#!/usr/bin/env python3
"""Run k-fold cross-validation for a fixed uplift champion.

This is an audit tool, not a model-selection loop. It keeps the champion
architecture and hyperparameters fixed, rotates validation folds over the
labeled training table, and writes standalone artifacts so final reports can
remain untouched until the audit is reviewed.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.uplift import (  # noqa: E402
    UpliftFeatureArtifact,
    UpliftProjectContract,
    UpliftSplitContract,
    UpliftTableSchema,
    UpliftTrialSpec,
)
from src.uplift.metrics import normalized_qini_auc_score  # noqa: E402
from src.uplift.templates import run_uplift_template  # noqa: E402


DEFAULT_FEATURE_METADATA = (
    ROOT
    / "artifacts"
    / "uplift"
    / "run_20260430_221602"
    / "features"
    / "uplift_features_train_9c740dc37c17.metadata.json"
)

CHAMPION_PARAMS = {
    "n_estimators": 400,
    "max_depth": 3,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 100,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_lambda": 10.0,
}


@dataclass(frozen=True)
class CrossValidationResult:
    output_dir: str
    summary_path: str
    metrics_path: str
    n_folds: int
    mean_qini_auc: float
    std_qini_auc: float
    mean_uplift_auc: float
    std_uplift_auc: float


def _build_contract(data_dir: Path) -> UpliftProjectContract:
    return UpliftProjectContract(
        task_name="retailhero-uplift",
        description="Fixed champion k-fold cross-validation audit.",
        table_schema=UpliftTableSchema(
            clients_table=str(data_dir / "clients.csv"),
            purchases_table=str(data_dir / "purchases.csv"),
            products_table=str(data_dir / "products.csv"),
            train_table=str(data_dir / "uplift_train.csv"),
            scoring_table=str(data_dir / "uplift_test.csv"),
        ),
        split_contract=UpliftSplitContract(),
        feature_sources=["clients", "purchases", "products"],
    )


def _labeled_feature_frame(
    contract: UpliftProjectContract,
    feature_artifact: UpliftFeatureArtifact,
) -> pd.DataFrame:
    features = pd.read_csv(feature_artifact.artifact_path)
    labels = pd.read_csv(
        contract.table_schema.train_table,
        usecols=[
            contract.entity_key,
            contract.treatment_column,
            contract.target_column,
        ],
    )
    return features.merge(labels, on=contract.entity_key, how="inner")


def _splitter(
    labeled: pd.DataFrame,
    *,
    contract: UpliftProjectContract,
    n_folds: int,
    seed: int,
):
    joint_key = (
        labeled[contract.treatment_column].astype(str)
        + "_"
        + labeled[contract.target_column].astype(str)
    )
    if int(joint_key.value_counts().min()) >= n_folds:
        return "joint_treatment_outcome", StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=seed,
        ).split(labeled, joint_key)

    treatment_key = labeled[contract.treatment_column].astype(str)
    if int(treatment_key.value_counts().min()) >= n_folds:
        return "treatment_only", StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=seed,
        ).split(labeled, treatment_key)

    return "random", KFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed,
    ).split(labeled)


def _champion_spec(feature_recipe_id: str, *, seed: int, spec_id: str) -> UpliftTrialSpec:
    return UpliftTrialSpec(
        spec_id=spec_id,
        hypothesis_id="cv_audit__RUN-c5e6e86f",
        template_name="class_transformation_lightgbm",
        learner_family="class_transformation",
        base_estimator="lightgbm",
        feature_recipe_id=feature_recipe_id,
        params=dict(CHAMPION_PARAMS),
        split_seed=seed,
    )


def run_cross_validation(
    contract: UpliftProjectContract,
    *,
    feature_artifact: UpliftFeatureArtifact,
    output_dir: str | Path,
    n_folds: int = 5,
    seed: int = 20260501,
) -> CrossValidationResult:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    labeled = _labeled_feature_frame(contract, feature_artifact)
    strategy, splits = _splitter(
        labeled,
        contract=contract,
        n_folds=n_folds,
        seed=seed,
    )

    rows: list[dict[str, Any]] = []
    for fold_idx, (train_idx, eval_idx) in enumerate(splits, start=1):
        fold_dir = output / f"fold_{fold_idx:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_df = labeled.iloc[train_idx].reset_index(drop=True)
        eval_df = labeled.iloc[eval_idx].reset_index(drop=True)
        spec = _champion_spec(
            feature_artifact.feature_recipe_id,
            seed=seed,
            spec_id=f"CV-{fold_idx:02d}-RUN-c5e6e86f",
        )
        result = run_uplift_template(
            spec,
            train_df=train_df,
            eval_df=eval_df,
            entity_key=contract.entity_key,
            treatment_col=contract.treatment_column,
            target_col=contract.target_column,
            cutoff_grid=contract.evaluation_policy.cutoff_grid,
        )
        predictions = result.predictions
        predictions.to_csv(fold_dir / "predictions.csv", index=False)
        result.decile_table.to_csv(fold_dir / "decile_table.csv", index=False)
        result.qini_curve.to_csv(fold_dir / "qini_curve.csv", index=False)
        result.uplift_curve.to_csv(fold_dir / "uplift_curve.csv", index=False)
        (fold_dir / "result_card.json").write_text(
            result.result_card.model_dump_json(indent=2),
            encoding="utf-8",
        )
        normalized_qini = normalized_qini_auc_score(
            predictions["target"].to_numpy(),
            predictions["treatment_flg"].to_numpy(),
            predictions["uplift"].to_numpy(),
        )
        row = {
            "fold": fold_idx,
            "n_train": int(len(train_df)),
            "n_eval": int(len(eval_df)),
            "treatment_rate_eval": float(eval_df[contract.treatment_column].mean()),
            "target_rate_eval": float(eval_df[contract.target_column].mean()),
            "qini_auc": result.result_card.qini_auc,
            "normalized_qini_auc": round(float(normalized_qini), 6),
            "uplift_auc": result.result_card.uplift_auc,
        }
        for key, value in result.result_card.uplift_at_k.items():
            row[f"uplift_{key}"] = value
        rows.append(row)

    metrics = pd.DataFrame(rows)
    metrics_path = output / "fold_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    summary = _summary_payload(
        metrics,
        contract=contract,
        feature_artifact=feature_artifact,
        strategy=strategy,
        n_folds=n_folds,
        seed=seed,
        output_dir=output,
        metrics_path=metrics_path,
    )
    summary_path = output / "cv_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (output / "CV_SUMMARY.md").write_text(_summary_markdown(summary, metrics), encoding="utf-8")
    return CrossValidationResult(
        output_dir=str(output),
        summary_path=str(summary_path),
        metrics_path=str(metrics_path),
        n_folds=n_folds,
        mean_qini_auc=summary["metrics"]["qini_auc"]["mean"],
        std_qini_auc=summary["metrics"]["qini_auc"]["std"],
        mean_uplift_auc=summary["metrics"]["uplift_auc"]["mean"],
        std_uplift_auc=summary["metrics"]["uplift_auc"]["std"],
    )


def _summary_payload(
    metrics: pd.DataFrame,
    *,
    contract: UpliftProjectContract,
    feature_artifact: UpliftFeatureArtifact,
    strategy: str,
    n_folds: int,
    seed: int,
    output_dir: Path,
    metrics_path: Path,
) -> dict[str, Any]:
    metric_columns = [
        column
        for column in metrics.columns
        if column
        not in {
            "fold",
            "n_train",
            "n_eval",
            "treatment_rate_eval",
            "target_rate_eval",
        }
    ]
    return {
        "champion_run_id": "RUN-c5e6e86f",
        "template_name": "class_transformation_lightgbm",
        "learner_family": "class_transformation",
        "base_estimator": "lightgbm",
        "params": CHAMPION_PARAMS,
        "feature_recipe_id": feature_artifact.feature_recipe_id,
        "feature_artifact_id": feature_artifact.feature_artifact_id,
        "feature_metadata_path": feature_artifact.metadata_path,
        "task_name": contract.task_name,
        "n_rows": int(metrics["n_eval"].sum()),
        "n_folds": n_folds,
        "seed": seed,
        "split_strategy": strategy,
        "selection_policy": "fixed champion only; no model selection inside CV",
        "interpretation_notes": [
            "Raw qini_auc depends on the evaluation fold size; compare normalized_qini_auc across split designs.",
            "K-fold CV uses 80% train / 20% eval per fold, unlike the original 70/15/15 train/validation/held-out split.",
            "CV metrics are a stability audit for a fixed champion and must not replace the held-out test comparison.",
        ],
        "output_dir": str(output_dir),
        "metrics_path": str(metrics_path),
        "metrics": {
            column: {
                "mean": round(float(metrics[column].mean()), 6),
                "std": round(float(metrics[column].std(ddof=1)), 6),
                "min": round(float(metrics[column].min()), 6),
                "max": round(float(metrics[column].max()), 6),
            }
            for column in metric_columns
        },
        "folds": metrics.to_dict(orient="records"),
    }


def _summary_markdown(summary: dict[str, Any], metrics: pd.DataFrame) -> str:
    lines = [
        "# Champion Cross-Validation Audit",
        "",
        "This audit keeps the final AutoLift champion fixed and rotates evaluation folds.",
        "It is not used for model selection.",
        "",
        f"- Champion: `{summary['champion_run_id']}`",
        f"- Template: `{summary['template_name']}`",
        f"- Folds: {summary['n_folds']}",
        f"- Split strategy: `{summary['split_strategy']}`",
        f"- Seed: `{summary['seed']}`",
        "",
        "## Interpretation Notes",
        "",
    ]
    for note in summary["interpretation_notes"]:
        lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## Aggregate Metrics",
            "",
            "| Metric | Mean | Std | Min | Max |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for metric, values in summary["metrics"].items():
        lines.append(
            f"| {metric} | {values['mean']:.6f} | {values['std']:.6f} | "
            f"{values['min']:.6f} | {values['max']:.6f} |"
        )
    lines.extend(["", "## Fold Metrics", ""])
    lines.extend(_metrics_markdown_table(metrics))
    lines.append("")
    return "\n".join(lines)


def _metrics_markdown_table(metrics: pd.DataFrame) -> list[str]:
    columns = list(metrics.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in metrics.to_dict(orient="records"):
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="retailhero-uplift/data")
    parser.add_argument("--feature-metadata", default=str(DEFAULT_FEATURE_METADATA))
    parser.add_argument(
        "--output-dir",
        default="artifacts/uplift/cv_audit_RUN-c5e6e86f",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260501)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    data_dir = _resolve(args.data_dir)
    feature_metadata = _resolve(args.feature_metadata)
    output_dir = _resolve(args.output_dir)
    contract = _build_contract(data_dir)
    feature_artifact = UpliftFeatureArtifact.model_validate_json(
        feature_metadata.read_text(encoding="utf-8")
    )
    result = run_cross_validation(
        contract,
        feature_artifact=feature_artifact,
        output_dir=output_dir,
        n_folds=args.folds,
        seed=args.seed,
    )
    print(
        "SUMMARY_JSON="
        + json.dumps(
            {
                "output_dir": result.output_dir,
                "summary_path": result.summary_path,
                "metrics_path": result.metrics_path,
                "n_folds": result.n_folds,
                "mean_qini_auc": result.mean_qini_auc,
                "std_qini_auc": result.std_qini_auc,
                "mean_uplift_auc": result.mean_uplift_auc,
                "std_uplift_auc": result.std_uplift_auc,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
