"""Dataset validation and split diagnostics for uplift experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

from src.models.uplift import UpliftProjectContract, UpliftSplitContract


class UpliftDatasetValidationReport(BaseModel):
    """Result of validating an uplift dataset against its contract."""

    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    table_rows: Dict[str, int] = Field(default_factory=dict)
    scoring_is_unlabeled: bool = False
    treatment_counts: Dict[int, int] = Field(default_factory=dict)
    target_counts: Dict[int, int] = Field(default_factory=dict)
    join_key_coverage: Dict[str, float] = Field(default_factory=dict)
    null_rate_warnings: Dict[str, Dict[str, float]] = Field(default_factory=dict)


class TreatmentControlBalanceDiagnostics(BaseModel):
    """Treatment/control balance summary over labeled uplift rows."""

    treatment_counts: Dict[int, int]
    target_rates_by_treatment: Dict[int, float]
    average_treatment_effect: float = 0.0
    joint_counts: Dict[str, int]
    standardized_mean_differences: Dict[str, float] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


@dataclass
class StratificationDecision:
    """Selected stratification strategy and optional key."""

    strategy: Literal["joint_treatment_outcome", "treatment_only", "random"]
    key: Optional[pd.Series]
    warnings: List[str] = field(default_factory=list)


def _read_csv(path: str | Path, *, nrows: Optional[int] = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, nrows=nrows)
    except FileNotFoundError:
        raise
    except Exception as exc:  # pragma: no cover - exact pandas errors vary by version
        raise ValueError(f"failed to read CSV {path}: {exc}") from exc


def _require_columns(
    df: pd.DataFrame,
    table_name: str,
    required: List[str],
    errors: List[str],
) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        errors.append(f"{table_name} missing required columns: {missing}")


def _binary_counts(
    series: pd.Series,
    column: str,
    errors: List[str],
) -> Dict[int, int]:
    values = set(series.dropna().unique().tolist())
    if not values.issubset({0, 1}):
        errors.append(f"{column} must be binary 0/1, found {sorted(values)}")
    return {int(k): int(v) for k, v in series.value_counts().sort_index().items()}


def _null_rate_warnings(
    df: pd.DataFrame,
    *,
    table_name: str,
    columns: List[str],
    threshold: float,
) -> Dict[str, float]:
    rates: Dict[str, float] = {}
    for col in columns:
        if col not in df.columns:
            continue
        rate = float(df[col].isna().mean())
        if rate > threshold:
            rates[col] = round(rate, 6)
    return rates


def _join_key_coverage(
    *,
    left_ids: pd.Series,
    right_ids: pd.Series,
) -> float:
    left = set(left_ids.dropna().astype(str).tolist())
    if not left:
        return 0.0
    right = set(right_ids.dropna().astype(str).tolist())
    return round(float(len(left.intersection(right))) / float(len(left)), 6)


def validate_uplift_dataset(
    contract: UpliftProjectContract,
    *,
    purchases_sample_rows: int = 1000,
    high_null_warning_threshold: float = 0.30,
) -> UpliftDatasetValidationReport:
    """Validate core RetailHero-style uplift table semantics."""
    errors: List[str] = []
    warnings: List[str] = []
    table_rows: Dict[str, int] = {}
    join_key_coverage: Dict[str, float] = {}
    null_rate_warnings: Dict[str, Dict[str, float]] = {}

    schema = contract.table_schema

    try:
        clients = _read_csv(schema.clients_table)
        train = _read_csv(schema.train_table)
        scoring = _read_csv(schema.scoring_table)
        purchases_sample = _read_csv(schema.purchases_table, nrows=purchases_sample_rows)
        products = _read_csv(schema.products_table) if schema.products_table else None
    except FileNotFoundError as exc:
        return UpliftDatasetValidationReport(
            valid=False,
            errors=[f"missing required table: {exc.filename}"],
        )
    except ValueError as exc:
        return UpliftDatasetValidationReport(valid=False, errors=[str(exc)])

    table_rows["clients"] = len(clients)
    table_rows["train"] = len(train)
    table_rows["scoring"] = len(scoring)
    table_rows["purchases_sample"] = len(purchases_sample)
    if products is not None:
        table_rows["products"] = len(products)

    _require_columns(clients, "clients", [contract.entity_key], errors)
    _require_columns(
        train,
        "train",
        [contract.entity_key, contract.treatment_column, contract.target_column],
        errors,
    )
    _require_columns(scoring, "scoring", [contract.entity_key], errors)
    _require_columns(
        purchases_sample,
        "purchases",
        [contract.entity_key, "transaction_id", "transaction_datetime", "purchase_sum"],
        errors,
    )

    # Null-rate warnings (mirrors phase-1 ingestion notebook signal).
    # These are warnings (not errors) because some columns are legitimately sparse
    # in some dataset variants; we still want the pipeline to proceed deterministically.
    null_rate_warnings["clients"] = _null_rate_warnings(
        clients,
        table_name="clients",
        columns=[contract.entity_key, "age", "gender", "first_issue_date"],
        threshold=high_null_warning_threshold,
    )
    null_rate_warnings["train"] = _null_rate_warnings(
        train,
        table_name="train",
        columns=[contract.entity_key, contract.treatment_column, contract.target_column, "first_issue_date"],
        threshold=high_null_warning_threshold,
    )
    null_rate_warnings["scoring"] = _null_rate_warnings(
        scoring,
        table_name="scoring",
        columns=[contract.entity_key],
        threshold=high_null_warning_threshold,
    )
    null_rate_warnings["purchases_sample"] = _null_rate_warnings(
        purchases_sample,
        table_name="purchases",
        columns=[contract.entity_key, "transaction_datetime", "purchase_sum"],
        threshold=high_null_warning_threshold,
    )
    if products is not None:
        null_rate_warnings["products"] = _null_rate_warnings(
            products,
            table_name="products",
            columns=["product_id"],
            threshold=high_null_warning_threshold,
        )
    null_rate_warnings = {k: v for k, v in null_rate_warnings.items() if v}
    if null_rate_warnings:
        warnings.append(
            f"high null rates detected (> {high_null_warning_threshold:.0%}) in: {sorted(null_rate_warnings)}"
        )

    if contract.target_column in scoring.columns:
        errors.append("scoring table must not contain target column")
    if contract.treatment_column in scoring.columns:
        errors.append("scoring table must not contain treatment column")
    scoring_is_unlabeled = (
        contract.target_column not in scoring.columns
        and contract.treatment_column not in scoring.columns
    )

    if contract.entity_key in train.columns:
        duplicated = int(train[contract.entity_key].duplicated().sum())
        if duplicated:
            errors.append(f"train has duplicate {contract.entity_key} rows: {duplicated}")

    if contract.entity_key in scoring.columns:
        duplicated = int(scoring[contract.entity_key].duplicated().sum())
        if duplicated:
            errors.append(
                f"scoring has duplicate {contract.entity_key} rows: {duplicated}"
            )

    if contract.entity_key in train.columns and contract.entity_key in scoring.columns:
        overlap = set(train[contract.entity_key]).intersection(scoring[contract.entity_key])
        if overlap:
            errors.append(f"train/scoring overlap detected: {len(overlap)} ids")

    # Join-key coverage diagnostics (mirrors phase-1 ingestion notebook).
    if contract.entity_key in clients.columns and contract.entity_key in train.columns:
        join_key_coverage["train_in_clients"] = _join_key_coverage(
            left_ids=train[contract.entity_key],
            right_ids=clients[contract.entity_key],
        )
    if contract.entity_key in clients.columns and contract.entity_key in scoring.columns:
        join_key_coverage["scoring_in_clients"] = _join_key_coverage(
            left_ids=scoring[contract.entity_key],
            right_ids=clients[contract.entity_key],
        )
    if contract.entity_key in clients.columns and contract.entity_key in purchases_sample.columns:
        join_key_coverage["purchases_sample_in_clients"] = _join_key_coverage(
            left_ids=purchases_sample[contract.entity_key],
            right_ids=clients[contract.entity_key],
        )

    # Date parsing sanity (warning-level; features.py will still coerce deterministically).
    if "transaction_datetime" in purchases_sample.columns:
        parsed = pd.to_datetime(purchases_sample["transaction_datetime"], errors="coerce")
        bad_rate = float(parsed.isna().mean())
        if bad_rate > 0.0:
            warnings.append(f"purchases_sample has unparseable transaction_datetime rows: {bad_rate:.2%}")
    if "first_issue_date" in train.columns:
        parsed = pd.to_datetime(train["first_issue_date"], errors="coerce")
        bad_rate = float(parsed.isna().mean())
        if bad_rate > 0.0:
            warnings.append(f"train has unparseable first_issue_date rows: {bad_rate:.2%}")

    treatment_counts: Dict[int, int] = {}
    target_counts: Dict[int, int] = {}
    if contract.treatment_column in train.columns:
        treatment_counts = _binary_counts(
            train[contract.treatment_column],
            contract.treatment_column,
            errors,
        )
    if contract.target_column in train.columns:
        target_counts = _binary_counts(train[contract.target_column], contract.target_column, errors)

    return UpliftDatasetValidationReport(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        table_rows=table_rows,
        scoring_is_unlabeled=scoring_is_unlabeled,
        treatment_counts=treatment_counts,
        target_counts=target_counts,
        join_key_coverage=join_key_coverage,
        null_rate_warnings=null_rate_warnings,
    )


def compute_treatment_control_balance(
    train_df: pd.DataFrame,
    *,
    entity_key: str,
    treatment_col: str,
    target_col: str,
    feature_df: Optional[pd.DataFrame] = None,
    numeric_columns: Optional[List[str]] = None,
    smd_warning_threshold: float = 0.1,
) -> TreatmentControlBalanceDiagnostics:
    """Compute simple treatment/control balance diagnostics."""
    treatment_counts = {
        int(k): int(v) for k, v in train_df[treatment_col].value_counts().sort_index().items()
    }
    target_rates = {
        int(k): float(v)
        for k, v in train_df.groupby(treatment_col)[target_col].mean().round(6).items()
    }
    average_treatment_effect = round(float(target_rates.get(1, 0.0) - target_rates.get(0, 0.0)), 6)
    joint_series = (
        train_df[treatment_col].astype(str) + ":" + train_df[target_col].astype(str)
    )
    joint_counts = {str(k): int(v) for k, v in joint_series.value_counts().sort_index().items()}

    smds: Dict[str, float] = {}
    warnings: List[str] = []
    if feature_df is not None and numeric_columns:
        merged = train_df[[entity_key, treatment_col]].merge(
            feature_df[[entity_key] + numeric_columns],
            on=entity_key,
            how="inner",
        )
        for col in numeric_columns:
            treated = merged.loc[merged[treatment_col] == 1, col].dropna()
            control = merged.loc[merged[treatment_col] == 0, col].dropna()
            pooled_var = (treated.var(ddof=0) + control.var(ddof=0)) / 2
            smd = 0.0 if pooled_var == 0 or pd.isna(pooled_var) else abs(
                float(treated.mean() - control.mean()) / float(pooled_var ** 0.5)
            )
            smds[col] = round(smd, 6)
            if smd > smd_warning_threshold:
                warnings.append(
                    f"standardized mean difference for {col} exceeds {smd_warning_threshold}: {smd:.3f}"
                )

    return TreatmentControlBalanceDiagnostics(
        treatment_counts=treatment_counts,
        target_rates_by_treatment=target_rates,
        average_treatment_effect=average_treatment_effect,
        joint_counts=joint_counts,
        standardized_mean_differences=smds,
        warnings=warnings,
    )


def _is_feasible(series: pd.Series, min_stratum_size: int) -> bool:
    counts = series.value_counts()
    return bool(not counts.empty and (counts >= min_stratum_size).all())


def determine_stratification(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    target_col: str,
    split_contract: UpliftSplitContract,
) -> StratificationDecision:
    """Choose joint, treatment-only, or random stratification deterministically."""
    warnings: List[str] = []

    joint_key = df[treatment_col].astype(str) + ":" + df[target_col].astype(str)
    if _is_feasible(joint_key, split_contract.min_stratum_size):
        return StratificationDecision(
            strategy="joint_treatment_outcome",
            key=joint_key,
            warnings=[],
        )

    warnings.append(
        "joint treatment/outcome stratification infeasible; trying treatment-only stratification"
    )
    treatment_key = df[treatment_col]
    if _is_feasible(treatment_key, split_contract.min_stratum_size):
        return StratificationDecision(
            strategy="treatment_only",
            key=treatment_key,
            warnings=warnings,
        )

    warnings.append("treatment-only stratification infeasible; falling back to random split")
    return StratificationDecision(strategy="random", key=None, warnings=warnings)
