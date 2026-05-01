"""
phase6_memory.py
================
Phase VI — Experiment Memory + Retry Control

Agents : ExperimentMemoryAgent, RetryControllerAgent
Inputs : TrainingResult (from phase3_execution), ExperimentMemory (from experiment_planning)
Outputs: logs/experiment_memory.jsonl  (append-only JSONL knowledge base)
         RetryDecision                 (should_continue, reason, suggested_next_action)
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from experiment_planning import ExperimentMemory, LLMClient, _parse_json
from phase3_execution import TrainingResult


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _params_hash(params: dict, learner_family: str, feature_recipe: str) -> str:
    """MD5 of the canonicalised (params, learner_family, feature_recipe) triple."""
    key = json.dumps(
        {"params": params, "learner_family": learner_family, "feature_recipe": feature_recipe},
        sort_keys=True,
    )
    return hashlib.md5(key.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RetryDecision:
    should_continue: bool
    reason: str
    suggested_next_action: str


# ---------------------------------------------------------------------------
# Agent 3: ExperimentMemoryAgent
# ---------------------------------------------------------------------------

class ExperimentMemoryAgent(ExperimentMemory):
    """
    Extends ExperimentMemory with the richer JSONL schema required by Phase III+.

    Record schema fields
    --------------------
    run_id, parent_run_id, hypothesis_id, stage_origin, dataset_version,
    feature_recipe, uplift_learner_family, base_estimator, params_hash,
    split_seed, qini, auuc, uplift_at_10, uplift_at_20,
    training_time_seconds, status, xai_summary, policy_summary,
    verdict, error_message, next_recommended_actions, timestamp

    Collision check: if the same params_hash + learner_family + feature_recipe
    already exists as a successful run, append_trial() warns and returns the
    prior record instead of writing a duplicate.
    """

    REQUIRED_FIELDS = {
        "run_id", "parent_run_id", "hypothesis_id", "stage_origin",
        "dataset_version", "feature_recipe", "uplift_learner_family",
        "base_estimator", "params_hash", "split_seed",
        "qini", "auuc", "uplift_at_10", "uplift_at_20",
        "training_time_seconds", "status", "verdict",
    }

    def __init__(self, path: str = "logs/experiment_memory.jsonl"):
        super().__init__(path=path)

    # ---- Write ----

    def append_trial(self, record: dict) -> dict | None:
        """
        Append one trial record.
        Returns a warning dict (with prior_result) if the same params were already
        run successfully; otherwise returns None and writes the record.
        """
        if "params_hash" not in record:
            record["params_hash"] = _params_hash(
                record.get("params", {}),
                record.get("uplift_learner_family", ""),
                record.get("feature_recipe", ""),
            )

        collision = self._find_collision(
            record["params_hash"],
            record.get("uplift_learner_family", ""),
            record.get("feature_recipe", ""),
        )
        if collision:
            warnings.warn(
                f"[ExperimentMemoryAgent] Duplicate params_hash {record['params_hash']!r} "
                f"already exists (trial {collision.get('run_id')!r}). "
                "Returning prior result instead of re-running.",
                UserWarning,
                stacklevel=2,
            )
            return {"warning": "duplicate_params", "prior_result": collision}

        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        return None

    # ---- Read ----

    def get_all(self) -> list[dict]:
        return self.read_all()

    def get_successful(self) -> list[dict]:
        return [r for r in self.read_all() if r.get("status") == "complete"]

    def get_best_by_metric(self, metric: str) -> Optional[dict]:
        successful = self.get_successful()
        if not successful:
            return None
        return max(successful, key=lambda r: r.get(metric, float("-inf")))

    def get_by_trial_id(self, trial_id: str) -> Optional[dict]:
        for r in self.read_all():
            if r.get("run_id") == trial_id:
                return r
        return None

    def get_benchmark_record(self) -> Optional[dict]:
        """Return the manual baseline record (stage_origin='manual_baseline'), if present."""
        for r in self.read_all():
            if r.get("stage_origin") == "manual_baseline":
                return r
        return None

    # ---- Internal ----

    def _find_collision(
        self, params_hash: str, learner_family: str, feature_recipe: str
    ) -> Optional[dict]:
        for r in self.read_all():
            if (
                r.get("params_hash") == params_hash
                and r.get("uplift_learner_family") == learner_family
                and r.get("feature_recipe") == feature_recipe
                and r.get("status") == "complete"
            ):
                return r
        return None

    # ---- Factory ----

    @staticmethod
    def record_from_result(
        result: TrainingResult,
        *,
        run_id: Optional[str] = None,
        parent_run_id: Optional[str] = None,
        hypothesis_id: Optional[str] = None,
        stage_origin: str = "training_execution",
        dataset_version: str = "x5_retailhero_v1",
        verdict: str = "pending",
        xai_summary: Optional[str] = None,
        policy_summary: Optional[str] = None,
        next_recommended_actions: Optional[list[str]] = None,
    ) -> dict:
        """Build a full new-schema record from a TrainingResult."""
        ph = _params_hash(result.params, result.learner_family, result.feature_recipe)
        return {
            "run_id":                   run_id or result.trial_id,
            "parent_run_id":            parent_run_id,
            "hypothesis_id":            hypothesis_id,
            "stage_origin":             stage_origin,
            "dataset_version":          dataset_version,
            "feature_recipe":           result.feature_recipe,
            "uplift_learner_family":    result.learner_family,
            "base_estimator":           result.base_estimator,
            "params_hash":              ph,
            "split_seed":               result.split_seed,
            "qini":                     result.metrics.get("qini"),
            "auuc":                     result.metrics.get("auuc"),
            "uplift_at_10":             result.metrics.get("uplift_at_10"),
            "uplift_at_20":             result.metrics.get("uplift_at_20"),
            "training_time_seconds":    result.training_time_seconds,
            "status":                   result.status,
            "xai_summary":              xai_summary,
            "policy_summary":           policy_summary,
            "verdict":                  verdict,
            "error_message":            result.error_message,
            "next_recommended_actions": next_recommended_actions,
            "timestamp":                datetime.utcnow().isoformat(),
        }


# ---------------------------------------------------------------------------
# Agent 4: RetryControllerAgent
# ---------------------------------------------------------------------------

_RETRY_SYSTEM_PROMPT = """You are an uplift modelling experiment controller.
Given the last 3 trial records, the current best AUUC, and a list of refuted
hypotheses, decide whether to run another trial.

Respond with JSON:
{
  "should_continue": true/false,
  "reason": "one sentence",
  "suggested_next_action": "brief description"
}
"""


class RetryControllerAgent:
    """
    Decides whether the pipeline should run another trial.

    Programmatic stops (checked first — no LLM involved):
    1. Total trial count ≥ MAX_TRIALS (budget exhausted)
    2. Last 3 successful trials have AUUC spread ≤ AUUC_FLATNESS_THRESHOLD (gains flattened)
    3. Last record has a params_hash that already ran successfully (duplicate)

    LLM fallback: used only when none of the above fire.
    """

    AUUC_FLATNESS_THRESHOLD = 0.002
    MAX_TRIALS = 20

    def __init__(self, memory: ExperimentMemoryAgent, llm: LLMClient):
        self.memory = memory
        self.llm    = llm

    def run(self) -> RetryDecision:
        all_records = self.memory.get_all()
        successful  = self.memory.get_successful()
        total_count = len(all_records)

        # ---- Programmatic stop 1: budget exhausted ----
        if total_count >= self.MAX_TRIALS:
            return RetryDecision(
                should_continue=False,
                reason=f"Budget exhausted: {total_count} total trials (max {self.MAX_TRIALS}).",
                suggested_next_action="Proceed to reporting phase.",
            )

        # ---- Programmatic stop 2: AUUC has flattened ----
        if len(successful) >= 3:
            recent_auuc = [r.get("auuc", 0.0) for r in successful[-3:]]
            if max(recent_auuc) - min(recent_auuc) <= self.AUUC_FLATNESS_THRESHOLD:
                return RetryDecision(
                    should_continue=False,
                    reason=(
                        f"AUUC has flattened: last 3 successful trials span "
                        f"{recent_auuc} (spread ≤ {self.AUUC_FLATNESS_THRESHOLD})."
                    ),
                    suggested_next_action="Proceed to reporting phase.",
                )

        # ---- Programmatic stop 3: duplicate params_hash ----
        if all_records:
            last       = all_records[-1]
            last_hash  = last.get("params_hash")
            if last_hash:
                prior = [
                    r for r in all_records[:-1]
                    if r.get("params_hash") == last_hash
                    and r.get("uplift_learner_family") == last.get("uplift_learner_family")
                    and r.get("feature_recipe") == last.get("feature_recipe")
                    and r.get("status") == "complete"
                ]
                if prior:
                    return RetryDecision(
                        should_continue=False,
                        reason="Duplicate params_hash — this configuration was already run successfully.",
                        suggested_next_action="Try a different learner family or feature recipe.",
                    )

        # ---- LLM fallback ----
        best_record = self.memory.get_best_by_metric("auuc")
        best_auuc   = best_record.get("auuc") if best_record else None
        refuted     = [
            r.get("hypothesis_id", "unknown")
            for r in all_records
            if r.get("verdict") == "refuted"
        ]
        last_3   = successful[-3:] if len(successful) >= 3 else successful
        user_msg = json.dumps({
            "last_3_trials":        last_3,
            "current_best_auuc":    best_auuc,
            "refuted_hypotheses":   refuted,
            "total_trials_so_far":  total_count,
        }, indent=2)

        raw    = self.llm.chat(_RETRY_SYSTEM_PROMPT, user_msg)
        parsed = _parse_json(raw)

        return RetryDecision(
            should_continue=bool(parsed.get("should_continue", True)),
            reason=parsed.get("reason", "LLM recommended continuing."),
            suggested_next_action=parsed.get("suggested_next_action", "Run next trial."),
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    import uuid
    import numpy as np
    import pandas as pd
    from experiment_planning import LLMClient
    from phase3_execution import (
        CodeConfigAgent, TrainingExecutionAgent, TrainingResult,
        VALID_LEARNER_FAMILIES, VALID_BASE_ESTIMATORS,
    )

    print("=" * 60)
    print("Phase VI smoke test — ExperimentMemoryAgent + RetryControllerAgent")
    print("=" * 60)

    RNG = np.random.default_rng(0)
    N   = 20

    stub_train = pd.DataFrame({
        "client_id":     [f"C{i:03d}" for i in range(N)],
        "target":        RNG.integers(0, 2, N).tolist(),
        "treatment_flg": [1] * (N // 2) + [0] * (N // 2),
    })
    stub_features = pd.DataFrame(RNG.random((N, 10)), columns=[f"f{i}" for i in range(10)])
    stub_features.insert(0, "customer_id", [f"C{i:03d}" for i in range(N)])

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        logs_dir = tmp / "logs"
        logs_dir.mkdir()

        feat_path = logs_dir / "feature_table.parquet"
        stub_features.to_parquet(feat_path, index=False)

        trial_id  = str(uuid.uuid4())
        spec_dict = {
            "trial_id":       trial_id,
            "learner_family": "SoloModel",
            "base_estimator": "XGBoost",
            "feature_recipe": "rfm_baseline",
            "params":         {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1},
            "split_seed":     42,
        }
        spec_path = logs_dir / "trial_spec.json"
        spec_path.write_text(json.dumps(spec_dict, indent=2))

        trials_dir = tmp / "trials" / trial_id
        trials_dir.mkdir(parents=True)
        meta_path  = trials_dir / "trial_meta.json"
        meta_path.write_text(json.dumps({"trial_id": trial_id}))

        config = CodeConfigAgent(trial_spec_path=str(spec_path)).run()
        result = TrainingExecutionAgent(
            feature_table_path=str(feat_path),
            trial_meta_path=str(meta_path),
            train_df=stub_train,
            config=config,
        ).run()

        print("\n[1/2] ExperimentMemoryAgent")
        mem = ExperimentMemoryAgent(path=str(logs_dir / "experiment_memory.jsonl"))

        # First write: mark complete so collision check can fire
        rec = ExperimentMemoryAgent.record_from_result(result, hypothesis_id="h1", verdict="pending")
        rec["status"] = "complete"
        assert mem.append_trial(rec) is None, "Unexpected collision on first write"

        # Second write with same params_hash should collide
        dup = mem.append_trial({**rec, "run_id": "dup-" + rec["run_id"]})
        assert dup is not None, "Expected collision on duplicate write"
        print(f"  Collision detection: OK")
        print(f"  get_all()       : {len(mem.get_all())} record(s)")
        print(f"  get_successful(): {len(mem.get_successful())} record(s)")

        print("\n[2/2] RetryControllerAgent")
        llm      = LLMClient(provider="stub")
        decision = RetryControllerAgent(memory=mem, llm=llm).run()
        print(f"  should_continue = {decision.should_continue}")
        print(f"  reason          = {decision.reason}")

        print("\n" + "=" * 60)
        print("Phase VI smoke test PASSED")
        print("=" * 60)
