"""
phase7_output.py
================
Phase VII — Output / Reporting + Orchestration

Agents     : ManualBenchmarkAgent, ReportingAgent, AutoLiftOrchestrator
Inputs     : feature_table.parquet, train_df, ExperimentMemoryAgent (Phase VI),
             ExperimentPlanningPhase (Phase II)
Outputs    : trials/manual_baseline/  (benchmark artefacts)
             logs/final_report.md
             returns (report_markdown, BenchmarkResult) from AutoLiftOrchestrator.run()
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from experiment_planning import (
    ExperimentMemory,
    ExperimentPlanningPhase,
    LLMClient,
    TrialRecord,
    TrialSpec,
)
from phase3_execution import (
    TrainingExecutionAgent,
    TrainingResult,
    VALID_LEARNER_FAMILIES,
    VALID_BASE_ESTIMATORS,
    CodeConfigAgent,
)
from phase6_memory import ExperimentMemoryAgent, RetryControllerAgent, _params_hash


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MANUAL_BASELINE_CONFIG: dict[str, Any] = {
    "trial_id":       "manual_baseline",
    "learner_family": "TwoModels",
    "base_estimator": "XGBoost",
    "feature_recipe": "rfm_baseline",
    "params": {
        "n_estimators":  100,
        "max_depth":     5,
        "learning_rate": 0.1,
    },
    "split_seed": 42,
    "hypothesis": "Manual human-designed RFM + TwoModels baseline",
}

_MANUAL_BASELINE_FEATURES = ["recency_days", "frequency", "monetary_total"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    auuc: float
    qini: float
    uplift_at_10: float
    uplift_at_20: float
    training_time_seconds: float


# ---------------------------------------------------------------------------
# Agent 5: ManualBenchmarkAgent
# ---------------------------------------------------------------------------

class ManualBenchmarkAgent:
    """
    Runs the fixed human-designed baseline:
      TwoModels + XGBoost, RFM features (recency_days, frequency, monetary_total),
      n_estimators=100, max_depth=5, learning_rate=0.1, split_seed=42.

    Uses TrainingExecutionAgent internally.
    Saves output artefacts to trials/manual_baseline/.
    """

    def __init__(
        self,
        feature_table_path: str,
        train_df: pd.DataFrame,
        trials_root: str = "trials",
    ):
        self.feature_table_path = Path(feature_table_path)
        self.train_df           = train_df.copy()
        self.trials_root        = Path(trials_root)

    def run(self) -> BenchmarkResult:
        baseline_dir = self.trials_root / "manual_baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)

        meta_path = baseline_dir / "trial_meta.json"
        with meta_path.open("w") as f:
            json.dump({**_MANUAL_BASELINE_CONFIG, "status": "pending"}, f, indent=2)

        # Keep only the three RFM columns that the baseline is defined on
        feature_df = pd.read_parquet(self.feature_table_path)
        if "client_id" in feature_df.columns and "customer_id" not in feature_df.columns:
            feature_df = feature_df.rename(columns={"client_id": "customer_id"})

        available_rfm = [c for c in _MANUAL_BASELINE_FEATURES if c in feature_df.columns]
        if not available_rfm:
            raise ValueError(
                f"Feature table is missing all RFM columns {_MANUAL_BASELINE_FEATURES}. "
                "Run the rfm_baseline feature recipe first."
            )

        rfm_path = baseline_dir / "rfm_features.parquet"
        feature_df[["customer_id"] + available_rfm].to_parquet(rfm_path, index=False)

        result = TrainingExecutionAgent(
            feature_table_path=str(rfm_path),
            trial_meta_path=str(meta_path),
            train_df=self.train_df,
            config=_MANUAL_BASELINE_CONFIG,
        ).run()

        if result.status == "failed":
            raise RuntimeError(
                f"ManualBenchmarkAgent training failed: {result.error_message}"
            )

        return BenchmarkResult(
            auuc=result.metrics.get("auuc", float("nan")),
            qini=result.metrics.get("qini", float("nan")),
            uplift_at_10=result.metrics.get("uplift_at_10", float("nan")),
            uplift_at_20=result.metrics.get("uplift_at_20", float("nan")),
            training_time_seconds=result.training_time_seconds,
        )


# ---------------------------------------------------------------------------
# Agent 6: ReportingAgent
# ---------------------------------------------------------------------------

class ReportingAgent:
    """
    Reads the full knowledge base via ExperimentMemoryAgent and produces
    logs/final_report.md.

    The report contains:
    - Champion model (trial_id, family, estimator, params)
    - Best metrics (AUUC, Qini, Uplift@10, Uplift@20)
    - Agent vs manual baseline comparison table
    - Supported / refuted hypotheses
    - Total trials, total training time
    - Recommended targeting policy
    """

    def __init__(
        self,
        memory: ExperimentMemoryAgent,
        benchmark: Optional[BenchmarkResult] = None,
        output_path: str = "logs/final_report.md",
    ):
        self.memory      = memory
        self.benchmark   = benchmark
        self.output_path = Path(output_path)

    def run(self) -> str:
        all_records      = self.memory.get_all()
        successful       = self.memory.get_successful()
        best             = self.memory.get_best_by_metric("auuc")
        total_trials     = len(all_records)
        total_train_time = sum(r.get("training_time_seconds", 0.0) for r in all_records)
        supported_hyp    = list({
            r.get("hypothesis_id", "") for r in all_records
            if r.get("verdict") == "supported" and r.get("hypothesis_id")
        })
        refuted_hyp      = list({
            r.get("hypothesis_id", "") for r in all_records
            if r.get("verdict") == "refuted" and r.get("hypothesis_id")
        })

        lines: list[str] = ["# AutoLift Experiment Report", ""]
        lines += [f"*Generated: {datetime.utcnow().isoformat()} UTC*", ""]

        # ---- Champion Model ----
        lines += ["## Champion Model", ""]
        if best:
            lines += [
                "| Field | Value |",
                "|-------|-------|",
                f"| Trial ID | `{best.get('run_id', 'N/A')}` |",
                f"| Learner Family | {best.get('uplift_learner_family', 'N/A')} |",
                f"| Base Estimator | {best.get('base_estimator', 'N/A')} |",
                f"| Feature Recipe | {best.get('feature_recipe', 'N/A')} |",
                f"| Split Seed | {best.get('split_seed', 'N/A')} |",
                "",
            ]
        else:
            lines += ["*No successful trials found.*", ""]

        # ---- Best Metrics ----
        lines += ["## Best Metrics", ""]
        if best:
            lines += [
                "| Metric | Value |",
                "|--------|-------|",
                f"| AUUC | {best.get('auuc', float('nan')):.4f} |",
                f"| Qini AUC | {best.get('qini', float('nan')):.4f} |",
                f"| Uplift@10% | {best.get('uplift_at_10', float('nan')):.4f} |",
                f"| Uplift@20% | {best.get('uplift_at_20', float('nan')):.4f} |",
                "",
            ]

        # ---- Agent vs Manual Baseline ----
        lines += ["## Agent vs Manual Baseline", ""]
        lines += [
            "| Metric | Agent Champion | Manual Baseline |",
            "|--------|---------------|-----------------|",
        ]
        for label, key in [
            ("AUUC", "auuc"), ("Qini AUC", "qini"),
            ("Uplift@10%", "uplift_at_10"), ("Uplift@20%", "uplift_at_20"),
        ]:
            agent_val = f"{best.get(key, float('nan')):.4f}" if best else "N/A"
            bench_val = (
                f"{getattr(self.benchmark, key, float('nan')):.4f}"
                if self.benchmark else "N/A"
            )
            lines.append(f"| {label} | {agent_val} | {bench_val} |")
        lines.append("")

        # ---- Hypotheses ----
        lines += ["## Supported Hypotheses", ""]
        lines += [f"- {h}" for h in supported_hyp] if supported_hyp else ["*None*"]
        lines.append("")

        lines += ["## Refuted Hypotheses", ""]
        lines += [f"- {h}" for h in refuted_hyp] if refuted_hyp else ["*None*"]
        lines.append("")

        # ---- Summary Stats ----
        lines += [
            "## Experiment Summary", "",
            "| Stat | Value |",
            "|------|-------|",
            f"| Total Trials | {total_trials} |",
            f"| Successful Trials | {len(successful)} |",
            f"| Total Training Time | {total_train_time:.1f}s |",
            "",
        ]

        # ---- Targeting Policy ----
        lines += ["## Recommended Targeting Policy", ""]
        if best:
            u10 = best.get("uplift_at_10")
            u20 = best.get("uplift_at_20")
            if u10 is not None and u20 is not None:
                if u10 >= u20:
                    lines += [
                        "Target the **top 10%** of customers by uplift score.",
                        f"Uplift@10% = {u10:.4f} vs Uplift@20% = {u20:.4f}.",
                        "Concentrating spend on the top decile yields the highest incremental return.",
                    ]
                else:
                    lines += [
                        "Target the **top 20%** of customers by uplift score.",
                        f"Uplift@20% = {u20:.4f} > Uplift@10% = {u10:.4f}.",
                        "Expanding to the top quintile captures additional incremental buyers.",
                    ]
            else:
                lines += ["*Insufficient metric data to recommend a targeting policy.*"]
        else:
            lines += ["*No champion model — cannot recommend a targeting policy.*"]

        report = "\n".join(lines) + "\n"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(report, encoding="utf-8")
        print(f"[ReportingAgent] Report saved to {self.output_path}")
        return report


# ---------------------------------------------------------------------------
# Orchestrator: AutoLiftOrchestrator
# ---------------------------------------------------------------------------

class AutoLiftOrchestrator:
    """
    Closes the planning → execution → retry loop end-to-end.

    Flow per iteration
    ------------------
    Phase II  : ExperimentPlanningPhase.run()        → TrialSpec
    Phase III : CodeConfigAgent + TrainingExecutionAgent → TrainingResult
    Phase VI  : ExperimentMemoryAgent.append_trial()
               RetryControllerAgent.run()            → RetryDecision

    After loop exits:
    Phase VII : ManualBenchmarkAgent (once, upfront)
               ReportingAgent                        → logs/final_report.md

    Schema bridge
    -------------
    ExperimentPlanningPhase reads the old ExperimentMemory schema
    (success, metrics.auuc, ...).  After each trial AutoLiftOrchestrator also
    writes a TrialRecord in the old schema to planner.memory so Phase II agents
    see prior results on every iteration without any changes to experiment_planning.py.
    """

    def __init__(
        self,
        planner: ExperimentPlanningPhase,
        feature_table_path: str,
        train_df: pd.DataFrame,
        logs_dir: str = "logs",
        trials_root: str = "trials",
        memory_agent: Optional[ExperimentMemoryAgent] = None,
        llm: Optional[LLMClient] = None,
        run_benchmark: bool = True,
    ):
        self.planner            = planner
        self.feature_table_path = Path(feature_table_path)
        self.train_df           = train_df.copy()
        self.logs_dir           = Path(logs_dir)
        self.trials_root        = Path(trials_root)
        self.llm                = llm or LLMClient(provider="stub")
        self.memory_agent       = memory_agent or ExperimentMemoryAgent(
            path=str(self.logs_dir / "experiment_memory.jsonl")
        )
        self.retry_controller   = RetryControllerAgent(
            memory=self.memory_agent, llm=self.llm
        )
        self.run_benchmark      = run_benchmark
        self._benchmark_result: Optional[BenchmarkResult] = None

    def run(
        self,
        max_iterations: int = 20,
        current_hypothesis: Optional[str] = None,
    ) -> tuple[str, Optional[BenchmarkResult]]:
        """
        Drive the experiment loop.
        Returns (report_markdown, benchmark_result).
        """
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.trials_root.mkdir(parents=True, exist_ok=True)

        # ---- Optional: one-time manual benchmark before the loop ----
        if self.run_benchmark and self._benchmark_result is None:
            print("[AutoLiftOrchestrator] Running ManualBenchmarkAgent...")
            try:
                bench_agent = ManualBenchmarkAgent(
                    feature_table_path=str(self.feature_table_path),
                    train_df=self.train_df,
                    trials_root=str(self.trials_root),
                )
                self._benchmark_result = bench_agent.run()
                self._write_benchmark_to_memory(self._benchmark_result)
                print(
                    f"  Baseline AUUC={self._benchmark_result.auuc:.4f}  "
                    f"Qini={self._benchmark_result.qini:.4f}"
                )
            except Exception as exc:
                print(f"  [WARN] ManualBenchmarkAgent failed: {exc}")

        # ---- Main retry loop ----
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            print(
                f"\n[AutoLiftOrchestrator] ===== Iteration {iteration}/{max_iterations} ====="
            )

            # Phase II: planning
            try:
                spec = self.planner.run(current_hypothesis=current_hypothesis)
            except Exception as exc:
                print(f"  [ERROR] ExperimentPlanningPhase failed: {exc}")
                break

            # Write trial_spec.json (CodeConfigAgent reads this)
            spec_dict = self._spec_to_dict(spec, split_seed=42 + iteration - 1)
            spec_path = self.logs_dir / "trial_spec.json"
            spec_path.write_text(json.dumps(spec_dict, indent=2))

            # Create trial directory + trial_meta.json
            trial_dir = self.trials_root / spec.trial_id
            trial_dir.mkdir(parents=True, exist_ok=True)
            meta_path = trial_dir / "trial_meta.json"
            meta_path.write_text(json.dumps(spec_dict, indent=2))

            # Phase III: validate config
            try:
                config = CodeConfigAgent(trial_spec_path=str(spec_path)).run()
            except ValueError as exc:
                print(f"  [ERROR] CodeConfigAgent rejected spec: {exc}")
                break

            # Phase III: train
            result = TrainingExecutionAgent(
                feature_table_path=str(self.feature_table_path),
                trial_meta_path=str(meta_path),
                train_df=self.train_df,
                config=config,
            ).run()

            print(f"  Trial {spec.trial_id[:8]}…  status={result.status}")
            if result.status == "complete":
                m = result.metrics
                print(
                    f"  AUUC={m['auuc']:.4f}  Qini={m['qini']:.4f}  "
                    f"Uplift@10={m['uplift_at_10']:.4f}  Uplift@20={m['uplift_at_20']:.4f}"
                )
            else:
                print(f"  error={result.error_message}")

            # Phase VI: write to new-schema memory
            verdict    = "inconclusive" if result.status == "complete" else "pending"
            new_record = ExperimentMemoryAgent.record_from_result(
                result,
                hypothesis_id=spec.trial_id,
                verdict=verdict,
                next_recommended_actions=[spec.stop_criteria],
            )
            collision = self.memory_agent.append_trial(new_record)
            if collision:
                print(
                    "  [WARN] Duplicate params_hash — same config already ran. "
                    "Advancing hypothesis."
                )
                current_hypothesis = None
                continue

            # Bridge: write old-schema record to planner.memory
            self._append_to_planner_memory(result, spec)

            # Carry hypothesis forward only on success
            current_hypothesis = spec.hypothesis if result.status == "complete" else None

            # Phase VI: retry decision
            decision = self.retry_controller.run()
            print(
                f"  RetryDecision: should_continue={decision.should_continue} — "
                f"{decision.reason}"
            )
            if not decision.should_continue:
                print(f"  Suggested next action: {decision.suggested_next_action}")
                break

        # Phase VII: report
        print("\n[AutoLiftOrchestrator] Generating final report...")
        reporter = ReportingAgent(
            memory=self.memory_agent,
            benchmark=self._benchmark_result,
            output_path=str(self.logs_dir / "final_report.md"),
        )
        report = reporter.run()
        return report, self._benchmark_result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _spec_to_dict(spec: TrialSpec, split_seed: int = 42) -> dict:
        """
        Serialise a TrialSpec to a JSON-compatible dict.
        TrialSpec stores learner_family and base_estimator only inside the
        free-text 'model' field (e.g. "SoloModel + XGBoost"), so we parse
        them out here to give CodeConfigAgent explicit fields to validate.
        """
        learner_family: Optional[str] = None
        for fam in VALID_LEARNER_FAMILIES:
            if fam in spec.model:
                learner_family = fam
                break
        base_estimator: Optional[str] = None
        for est in VALID_BASE_ESTIMATORS:
            if est in spec.model:
                base_estimator = est
                break
        return {
            "trial_id":              spec.trial_id,
            "learner_family":        learner_family or "SoloModel",
            "base_estimator":        base_estimator or "XGBoost",
            "feature_recipe":        spec.feature_recipe,
            "params":                spec.params,
            "split_seed":            split_seed,
            "hypothesis":            spec.hypothesis,
            "hypothesis_id":         spec.trial_id,
            "model":                 spec.model,
            "changes_from_previous": spec.changes_from_previous,
            "expected_improvement":  spec.expected_improvement,
            "stop_criteria":         spec.stop_criteria,
        }

    def _append_to_planner_memory(self, result: TrainingResult, spec: TrialSpec) -> None:
        """
        Write a TrialRecord (old ExperimentMemory schema) to planner.memory so
        Phase II agents can see results from the current run on the next iteration.
        """
        old_record = TrialRecord(
            trial_id=result.trial_id,
            timestamp=datetime.utcnow().isoformat(),
            learner_family=result.learner_family,
            base_estimator=result.base_estimator,
            feature_recipe=result.feature_recipe,
            hyperparams=result.params,
            metrics={
                "auuc":        result.metrics.get("auuc", 0.0),
                "qini":        result.metrics.get("qini", 0.0),
                "uplift_at_10": result.metrics.get("uplift_at_10", 0.0),
                "uplift_at_20": result.metrics.get("uplift_at_20", 0.0),
                "train_time":  result.training_time_seconds,
            },
            hypothesis=spec.hypothesis,
            hypothesis_status="inconclusive" if result.status == "complete" else "pending",
            success=(result.status == "complete"),
            error_notes=result.error_message or "",
        )
        self.planner.memory.append(old_record)

    def _write_benchmark_to_memory(self, bench: BenchmarkResult) -> None:
        """
        Persist the ManualBenchmarkAgent result to ExperimentMemoryAgent with
        stage_origin='manual_baseline' so ReportingAgent and future sessions
        can retrieve it via memory_agent.get_benchmark_record().
        """
        record = {
            "run_id":                   "manual_baseline",
            "parent_run_id":            None,
            "hypothesis_id":            "manual_baseline",
            "stage_origin":             "manual_baseline",
            "dataset_version":          "x5_retailhero_v1",
            "feature_recipe":           "rfm_baseline",
            "uplift_learner_family":    "TwoModels",
            "base_estimator":           "XGBoost",
            "params_hash":              _params_hash(
                _MANUAL_BASELINE_CONFIG["params"], "TwoModels", "rfm_baseline"
            ),
            "split_seed":               42,
            "qini":                     bench.qini,
            "auuc":                     bench.auuc,
            "uplift_at_10":             bench.uplift_at_10,
            "uplift_at_20":             bench.uplift_at_20,
            "training_time_seconds":    bench.training_time_seconds,
            "status":                   "complete",
            "verdict":                  "baseline",
            "xai_summary":              None,
            "policy_summary":           None,
            "error_message":            None,
            "next_recommended_actions": None,
            "timestamp":                datetime.utcnow().isoformat(),
        }
        # Write directly — bypass collision check since the benchmark params_hash is fixed
        with self.memory_agent.path.open("a") as f:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    import uuid
    import numpy as np
    from experiment_planning import ExperimentMemory as _EM, LLMClient

    print("=" * 65)
    print("Phase VII smoke test — AutoLiftOrchestrator (2 iterations, stub LLM)")
    print("=" * 65)

    RNG = np.random.default_rng(0)
    N   = 20

    stub_train = pd.DataFrame({
        "client_id":     [f"C{i:03d}" for i in range(N)],
        "target":        RNG.integers(0, 2, N).tolist(),
        "treatment_flg": [1] * (N // 2) + [0] * (N // 2),
    })
    stub_features = pd.DataFrame(RNG.random((N, 10)), columns=[f"f{i}" for i in range(10)])
    stub_features.insert(0, "customer_id", [f"C{i:03d}" for i in range(N)])
    stub_features["recency_days"]   = RNG.integers(1, 365, N)
    stub_features["frequency"]      = RNG.integers(1, 50, N)
    stub_features["monetary_total"] = RNG.random(N) * 5000.0

    with tempfile.TemporaryDirectory() as tmp:
        tmp       = Path(tmp)
        logs_dir  = tmp / "logs"
        logs_dir.mkdir()
        trials_dir = tmp / "trials"

        feat_path = logs_dir / "feature_table.parquet"
        stub_features.to_parquet(feat_path, index=False)

        # Build planner with its own old-schema memory
        planner_memory = _EM(path=str(logs_dir / "knowledge_base.jsonl"))
        llm            = LLMClient(provider="stub")
        planner        = ExperimentPlanningPhase(memory=planner_memory, llm=llm)

        mem2 = ExperimentMemoryAgent(path=str(logs_dir / "experiment_memory.jsonl"))

        orch = AutoLiftOrchestrator(
            planner=planner,
            feature_table_path=str(feat_path),
            train_df=stub_train,
            logs_dir=str(logs_dir),
            trials_root=str(trials_dir),
            memory_agent=mem2,
            llm=llm,
            run_benchmark=False,  # ManualBenchmarkAgent requires > 20 rows for a valid split
        )

        report, bench = orch.run(max_iterations=2)

        all_records = mem2.get_all()
        print(f"\n  Records in memory    : {len(all_records)}")
        print(f"  Planner memory rows  : {len(planner_memory.read_all())}")
        print(f"  Report length        : {len(report)} chars")

        assert "Champion Model" in report,   "Report missing 'Champion Model'"
        assert "Manual Baseline" in report,  "Report missing comparison table"
        assert len(all_records) >= 1,        "No records written after 2 iterations"
        assert len(planner_memory.read_all()) >= 1, "Old-schema memory not updated"

        print("\n" + "=" * 65)
        print("Phase VII smoke test PASSED")
        print("=" * 65)
