from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from src.models.task import RunConfig, ExperimentPlan


class ModelEntry(BaseModel):
    """One row of AutoGluon's leaderboard — a single trained model's performance.

    Created by: ResultParser, which iterates over the leaderboard DataFrame after fit().
    Stored in: RunResult.leaderboard. Read by: RefinerAgent to see which model families
    actually trained and how they ranked.
    """
    model_name: str
    score_val: float
    fit_time: float
    pred_time: float
    stack_level: int = 1
    score_train: Optional[float] = None  # only populated when leaderboard(extra_info=True) succeeds


class RunResult(BaseModel):
    """Raw output of one AutoGluon fit() call — exactly what the execution layer returned.

    Created by: ResultParser.from_predictor (success) or ResultParser.from_error (failure).
    Stored in: ExperimentRun.result.
    Read by: session.execute_node (accept/reject decision), RefinerAgent (leaderboard context).
    """
    status: Literal["success", "failed"]
    primary_metric: Optional[float] = None
    leaderboard: List[ModelEntry] = Field(default_factory=list)
    best_model_name: Optional[str] = None
    fit_time_seconds: float = 0.0
    error: Optional[str] = None


class DataProfile(BaseModel):
    """Statistical snapshot of the dataset, computed once at session start.

    Created by: session.py reading the CSV before any runs begin.
    Used by: IdeatorAgent (to generate dataset-aware hypotheses), Distiller (to compute
    TaskTraits for CaseEntry), and logged in session.log for human review.
    """
    n_rows: int
    n_features: int
    feature_types: Dict[str, int] = Field(default_factory=dict)
    target_distribution: Dict[str, Any] = Field(default_factory=dict)
    class_balance_ratio: float = 1.0
    missing_rate: float = 0.0
    high_cardinality_cols: List[str] = Field(default_factory=list)
    suspected_leakage_cols: List[str] = Field(default_factory=list)
    summary: str = ""


class RunDiagnostics(BaseModel):
    """Observations computed by session.py after a run completes — not from AutoGluon directly.

    Created by: session.execute_node after each run (overfitting_gap from ResultParser tuple,
    metric_vs_parent by comparing to parent node's metric).
    Stored in: ExperimentRun.diagnostics.
    Read by: RefinerAgent to decide whether to reduce complexity or diversify model families.
    All fields optional — diagnostics enrich agent decisions but are never required for correctness.
    """
    overfitting_gap: Optional[float] = None   # score_train - score_val; positive = overfitting
    metric_vs_parent: Optional[float] = None  # positive = improvement over incumbent
    failure_mode: Optional[str] = None


class ExperimentRun(BaseModel):
    """Complete record of one experiment run — the unit stored in decisions.jsonl.

    Created by: session.execute_node after each AutoGluon fit.
    Stored in: RunStore (appended to decisions.jsonl), ExperimentNode.entry.
    Read by: RefinerAgent and SelectorAgent (prior_runs history), Distiller (to build CaseEntry),
    RunStore.get_incumbent (to find the current best run).

    Composite of: what the agent planned (plan), how it was translated to AutoGluon kwargs (config),
    what AutoGluon returned (result), and what session.py computed afterward (diagnostics).
    """
    run_id: str
    node_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    config: RunConfig
    result: RunResult
    diagnostics: RunDiagnostics = Field(default_factory=RunDiagnostics)
    plan: Optional[ExperimentPlan] = None
    agent_rationale: str = ""
