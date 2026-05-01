"""EDA agent for uplift datasets.

The agent has two layers:

1. Deterministic profiling of the contract tables.
2. Optional LLM drafting of candidate uplift/segmentation hypotheses from the
   deterministic findings. The LLM output is advisory only and does not create
   executable trial specs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from src.models.uplift import UpliftProjectContract
from src.uplift.llm_client import ChatLLM


_SKILL_PATH = Path(__file__).parent / "skills" / "eda_hypothesis.md"


class UpliftEDAFinding(BaseModel):
    """One deterministic EDA finding."""

    topic: str
    finding: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class UpliftEDAHypothesisDraft(BaseModel):
    """One LLM-drafted hypothesis grounded in EDA evidence."""

    hypothesis: str
    rationale: str
    suggested_features: list[str] = Field(default_factory=list)
    segment_idea: str = ""
    risk_or_guardrail: str = ""


class UpliftEDAReport(BaseModel):
    """Combined deterministic EDA report and advisory hypothesis draft."""

    table_rows: dict[str, int]
    table_columns: dict[str, list[str]]
    null_rates: dict[str, dict[str, float]] = Field(default_factory=dict)
    treatment_counts: dict[int, int] = Field(default_factory=dict)
    target_counts: dict[int, int] = Field(default_factory=dict)
    target_rate_by_treatment: dict[int, float] = Field(default_factory=dict)
    average_treatment_effect: float | None = None
    purchase_summary: dict[str, float] = Field(default_factory=dict)
    segment_response_candidates: list[dict[str, Any]] = Field(default_factory=list)
    findings: list[UpliftEDAFinding] = Field(default_factory=list)
    llm_summary: str = ""
    drafted_hypotheses: list[UpliftEDAHypothesisDraft] = Field(default_factory=list)
    recommended_next_checks: list[str] = Field(default_factory=list)


class UpliftEDAAgent:
    """Profile an uplift dataset and draft advisory hypotheses."""

    def __init__(
        self,
        contract: UpliftProjectContract,
        llm: ChatLLM | None = None,
        *,
        purchases_sample_rows: int = 100_000,
    ) -> None:
        self.contract = contract
        self.llm = llm
        self.purchases_sample_rows = purchases_sample_rows

    def run(self) -> UpliftEDAReport:
        schema = self.contract.table_schema
        clients = pd.read_csv(schema.clients_table)
        train = pd.read_csv(schema.train_table)
        scoring = pd.read_csv(schema.scoring_table)
        purchases = pd.read_csv(schema.purchases_table, nrows=self.purchases_sample_rows)
        products = (
            pd.read_csv(schema.products_table)
            if schema.products_table and Path(schema.products_table).exists()
            else None
        )

        frames: dict[str, pd.DataFrame] = {
            "clients": clients,
            "train": train,
            "scoring": scoring,
            "purchases_sample": purchases,
        }
        if products is not None:
            frames["products"] = products

        report = UpliftEDAReport(
            table_rows={name: int(len(df)) for name, df in frames.items()},
            table_columns={name: list(df.columns) for name, df in frames.items()},
            null_rates=_null_rates(frames),
        )
        self._add_experiment_summary(report, train)
        self._add_purchase_summary(report, purchases)
        self._add_segment_candidates(report, clients, train)
        self._add_findings(report)
        self._draft_hypotheses(report)
        return report

    def _add_experiment_summary(
        self,
        report: UpliftEDAReport,
        train: pd.DataFrame,
    ) -> None:
        treatment = train[self.contract.treatment_column].astype(int)
        target = train[self.contract.target_column].astype(int)
        report.treatment_counts = {
            int(k): int(v) for k, v in treatment.value_counts().sort_index().items()
        }
        report.target_counts = {
            int(k): int(v) for k, v in target.value_counts().sort_index().items()
        }
        rates = train.groupby(self.contract.treatment_column)[self.contract.target_column].mean()
        report.target_rate_by_treatment = {
            int(k): round(float(v), 6) for k, v in rates.sort_index().items()
        }
        if {0, 1}.issubset(set(report.target_rate_by_treatment)):
            report.average_treatment_effect = round(
                report.target_rate_by_treatment[1] - report.target_rate_by_treatment[0],
                6,
            )

    def _add_purchase_summary(
        self,
        report: UpliftEDAReport,
        purchases: pd.DataFrame,
    ) -> None:
        summary: dict[str, float] = {}
        if self.contract.entity_key in purchases.columns:
            summary["sample_unique_customers"] = float(purchases[self.contract.entity_key].nunique())
        if "transaction_id" in purchases.columns:
            summary["sample_transactions"] = float(purchases["transaction_id"].nunique())
        else:
            summary["sample_transactions"] = float(len(purchases))
        if "purchase_sum" in purchases.columns:
            spend = pd.to_numeric(purchases["purchase_sum"], errors="coerce")
            summary.update(
                {
                    "purchase_sum_mean": round(float(spend.mean()), 6),
                    "purchase_sum_median": round(float(spend.median()), 6),
                    "purchase_sum_p95": round(float(spend.quantile(0.95)), 6),
                }
            )
        if "product_id" in purchases.columns:
            summary["sample_unique_products"] = float(purchases["product_id"].nunique())
        if "store_id" in purchases.columns:
            summary["sample_unique_stores"] = float(purchases["store_id"].nunique())
        report.purchase_summary = summary

    def _add_segment_candidates(
        self,
        report: UpliftEDAReport,
        clients: pd.DataFrame,
        train: pd.DataFrame,
    ) -> None:
        joined = train.merge(clients, on=self.contract.entity_key, how="left")
        candidates: list[dict[str, Any]] = []
        if "gender" in joined.columns:
            candidates.extend(
                _segment_response_table(
                    joined,
                    segment_col="gender",
                    treatment_col=self.contract.treatment_column,
                    target_col=self.contract.target_column,
                )
            )
        if "age" in joined.columns:
            age = pd.to_numeric(joined["age"], errors="coerce")
            if age.notna().nunique() >= 3:
                age_bins = pd.cut(
                    age,
                    bins=[0, 25, 35, 45, 55, 65, 200],
                    include_lowest=True,
                    duplicates="drop",
                )
                age_frame = joined.assign(age_band=age_bins.astype(str))
                candidates.extend(
                    _segment_response_table(
                        age_frame,
                        segment_col="age_band",
                        treatment_col=self.contract.treatment_column,
                        target_col=self.contract.target_column,
                    )
                )
        report.segment_response_candidates = sorted(
            candidates,
            key=lambda row: abs(float(row.get("response_gap", 0.0))),
            reverse=True,
        )[:10]

    def _add_findings(self, report: UpliftEDAReport) -> None:
        findings: list[UpliftEDAFinding] = []
        findings.append(
            UpliftEDAFinding(
                topic="dataset_shape",
                finding=(
                    f"Dataset contains {report.table_rows.get('train', 0)} labeled rows, "
                    f"{report.table_rows.get('scoring', 0)} scoring rows, and "
                    f"{report.table_rows.get('purchases_sample', 0)} sampled purchase rows."
                ),
                evidence=report.table_rows,
            )
        )
        if report.average_treatment_effect is not None:
            findings.append(
                UpliftEDAFinding(
                    topic="experiment_response",
                    finding=(
                        "Observed treatment-control target-rate difference is "
                        f"{report.average_treatment_effect:.4f}; uplift modeling should "
                        "focus on heterogeneous response rather than only average lift."
                    ),
                    evidence={
                        "target_rate_by_treatment": report.target_rate_by_treatment,
                        "average_treatment_effect": report.average_treatment_effect,
                    },
                )
            )
        if report.purchase_summary:
            findings.append(
                UpliftEDAFinding(
                    topic="transaction_history",
                    finding=(
                        "Purchase history is available for behavioral features such as "
                        "frequency, spend, product breadth, store breadth, and recency windows."
                    ),
                    evidence=report.purchase_summary,
                )
            )
        if report.segment_response_candidates:
            top = report.segment_response_candidates[0]
            findings.append(
                UpliftEDAFinding(
                    topic="candidate_segments",
                    finding=(
                        f"Segment '{top['segment']}' in {top['segment_col']} has the largest "
                        f"descriptive treatment-control response gap ({top['response_gap']:.4f}) "
                        "among checked demographic slices."
                    ),
                    evidence=top,
                )
            )
        for table, rates in report.null_rates.items():
            high = {col: rate for col, rate in rates.items() if rate > 0.30}
            if high:
                findings.append(
                    UpliftEDAFinding(
                        topic="missingness",
                        finding=f"{table} has high-null columns that may need explicit indicators.",
                        evidence=high,
                    )
                )
        report.findings = findings

    def _draft_hypotheses(self, report: UpliftEDAReport) -> None:
        if self.llm is None:
            return
        payload = {
            "table_rows": report.table_rows,
            "treatment_counts": report.treatment_counts,
            "target_counts": report.target_counts,
            "target_rate_by_treatment": report.target_rate_by_treatment,
            "average_treatment_effect": report.average_treatment_effect,
            "purchase_summary": report.purchase_summary,
            "segment_response_candidates": report.segment_response_candidates[:5],
            "findings": [finding.model_dump() for finding in report.findings],
            "constraints": {
                "no_performance_claims": True,
                "draft_only": True,
                "task": "predict uplift score and segment customers",
            },
        }
        try:
            parsed = _parse_json_object(
                self.llm(_load_skill(), json.dumps(payload, sort_keys=True))
            )
        except ValueError as exc:
            report.recommended_next_checks.append(f"EDA LLM draft unavailable: {exc}")
            return
        report.llm_summary = str(parsed.get("summary") or "")
        hypotheses = parsed.get("hypotheses", [])
        if isinstance(hypotheses, list):
            report.drafted_hypotheses = [
                UpliftEDAHypothesisDraft(
                    hypothesis=str(item.get("hypothesis") or ""),
                    rationale=str(item.get("rationale") or ""),
                    suggested_features=[
                        str(feature)
                        for feature in item.get("suggested_features", [])
                        if feature is not None
                    ]
                    if isinstance(item, dict)
                    else [],
                    segment_idea=str(item.get("segment_idea") or ""),
                    risk_or_guardrail=str(item.get("risk_or_guardrail") or ""),
                )
                for item in hypotheses
                if isinstance(item, dict) and item.get("hypothesis")
            ][:5]
        checks = parsed.get("recommended_next_checks", [])
        if isinstance(checks, list):
            report.recommended_next_checks.extend(str(check) for check in checks[:8])


def run_eda_phase(
    contract: UpliftProjectContract,
    llm: ChatLLM | None,
    *,
    output_dir: str | Path | None = None,
    purchases_sample_rows: int = 100_000,
) -> UpliftEDAReport:
    """Run the EDA agent and optionally persist JSON/Markdown artifacts."""
    report = UpliftEDAAgent(
        contract,
        llm,
        purchases_sample_rows=purchases_sample_rows,
    ).run()
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "eda_report.json").write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (output / "eda_report.md").write_text(
            render_eda_markdown(report),
            encoding="utf-8",
        )
    return report


def render_eda_markdown(report: UpliftEDAReport) -> str:
    """Render a concise human-readable EDA report."""
    lines = ["# EDA Agent Report", "", "## Dataset", ""]
    for table, rows in report.table_rows.items():
        n_cols = len(report.table_columns.get(table, []))
        lines.append(f"- {table}: {rows} rows, {n_cols} columns")
    lines.extend(["", "## Deterministic Findings", ""])
    for finding in report.findings:
        lines.append(f"- **{finding.topic}:** {finding.finding}")
    if report.llm_summary:
        lines.extend(["", "## LLM Hypothesis Draft", "", report.llm_summary, ""])
    if report.drafted_hypotheses:
        for idx, hypothesis in enumerate(report.drafted_hypotheses, start=1):
            lines.append(f"{idx}. {hypothesis.hypothesis}")
            if hypothesis.rationale:
                lines.append(f"   Rationale: {hypothesis.rationale}")
            if hypothesis.suggested_features:
                lines.append(
                    "   Suggested features: "
                    + ", ".join(hypothesis.suggested_features)
                )
            if hypothesis.segment_idea:
                lines.append(f"   Segment idea: {hypothesis.segment_idea}")
            if hypothesis.risk_or_guardrail:
                lines.append(f"   Guardrail: {hypothesis.risk_or_guardrail}")
    if report.recommended_next_checks:
        lines.extend(["", "## Recommended Next Checks", ""])
        for check in report.recommended_next_checks:
            lines.append(f"- {check}")
    lines.append("")
    return "\n".join(lines)


def _null_rates(frames: dict[str, pd.DataFrame]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name, frame in frames.items():
        rates = {
            str(col): round(float(rate), 6)
            for col, rate in frame.isna().mean().items()
            if float(rate) > 0
        }
        if rates:
            result[name] = rates
    return result


def _segment_response_table(
    frame: pd.DataFrame,
    *,
    segment_col: str,
    treatment_col: str,
    target_col: str,
    min_rows: int = 2,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment, group in frame.groupby(segment_col, dropna=False):
        if len(group) < min_rows:
            continue
        rates = group.groupby(treatment_col)[target_col].mean()
        counts = group[treatment_col].value_counts()
        if not {0, 1}.issubset(set(rates.index)):
            continue
        rows.append(
            {
                "segment_col": segment_col,
                "segment": str(segment),
                "n_rows": int(len(group)),
                "treated_rows": int(counts.get(1, 0)),
                "control_rows": int(counts.get(0, 0)),
                "treated_target_rate": round(float(rates.loc[1]), 6),
                "control_target_rate": round(float(rates.loc[0]), 6),
                "response_gap": round(float(rates.loc[1] - rates.loc[0]), 6),
            }
        )
    return rows


def _load_skill() -> str:
    if not _SKILL_PATH.exists():
        raise FileNotFoundError(f"Required EDA skill prompt is missing: {_SKILL_PATH}")
    return _SKILL_PATH.read_text(encoding="utf-8")


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise ValueError(f"no JSON object found in LLM output: {exc}") from exc
        payload = json.loads(match.group())
    if not isinstance(payload, dict):
        raise ValueError("LLM JSON payload must be an object")
    return payload
