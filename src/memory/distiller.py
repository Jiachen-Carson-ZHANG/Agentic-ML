from __future__ import annotations
import json
import uuid
from pathlib import Path
from typing import List
from src.llm.backend import LLMBackend, Message
from src.models.task import TaskSpec, ExperimentPlan
from src.models.results import DataProfile, RunEntry
from src.models.nodes import (
    CaseEntry, TaskTraits, WhatWorked, WhatFailed, SessionTrajectory,
)
from src.memory.trait_utils import rows_bucket, features_bucket, balance_bucket


class Distiller:
    """Summarises a completed session into a CaseEntry via LLM."""

    def __init__(self, llm: LLMBackend, prompt_path: str = "prompts/distiller.md") -> None:
        self._llm = llm
        self._system_prompt = Path(prompt_path).read_text()

    def distill(
        self,
        task: TaskSpec,
        data_profile: DataProfile,
        run_history: List[RunEntry],
    ) -> CaseEntry:
        successful = [
            r for r in run_history
            if r.result.status == "success" and r.result.primary_metric is not None
        ]
        best = max(successful, key=lambda r: r.result.primary_metric) if successful else None
        metrics = [r.result.primary_metric for r in successful]

        traits = TaskTraits(
            task_type=task.task_type,
            n_rows_bucket=rows_bucket(data_profile.n_rows),
            n_features_bucket=features_bucket(data_profile.n_features),
            class_balance=balance_bucket(data_profile.class_balance_ratio),
            feature_types=data_profile.feature_types,
        )
        trajectory = SessionTrajectory(
            n_runs=len(run_history),
            total_time_seconds=sum(r.result.fit_time_seconds for r in run_history),
            metric_progression=metrics,
        )

        user_msg = self._build_user_message(task, data_profile, run_history, metrics)
        response = self._llm.complete(
            messages=[
                Message(role="system", content=self._system_prompt),
                Message(role="user", content=user_msg),
            ],
            temperature=0.2,
        )

        parsed = json.loads(response)
        ww_raw = parsed.get("what_worked", {})
        wf_raw = parsed.get("what_failed", {})
        traj_raw = parsed.get("trajectory", {})

        # Reconstruct best ExperimentPlan from winning run's autogluon_kwargs
        if best:
            kw = best.config.autogluon_kwargs
            best_plan = ExperimentPlan(
                eval_metric=kw.get("eval_metric", task.eval_metric),
                model_families=list(kw.get("hyperparameters", {}).keys()),
                presets=kw.get("presets", "medium_quality"),
                time_limit=kw.get("time_limit", 120),
                feature_policy={},
                validation_policy={"holdout_frac": 0.2},
            )
            best_metric = best.result.primary_metric
        else:
            best_plan = ExperimentPlan(
                eval_metric=task.eval_metric, model_families=[],
                presets="medium_quality", time_limit=120,
                feature_policy={}, validation_policy={"holdout_frac": 0.2},
            )
            best_metric = 0.0

        what_worked = WhatWorked(
            best_config=best_plan,
            best_metric=best_metric,
            key_decisions=ww_raw.get("key_decisions", []),
            important_features=ww_raw.get("important_features", []),
            effective_presets=ww_raw.get("effective_presets", "medium_quality"),
        )
        what_failed = WhatFailed(
            failed_approaches=wf_raw.get("failed_approaches", []),
            failure_patterns=wf_raw.get("failure_patterns", []),
        )
        trajectory.turning_points = traj_raw.get("turning_points", [])

        return CaseEntry(
            case_id=str(uuid.uuid4()),
            task_traits=traits,
            what_worked=what_worked,
            what_failed=what_failed,
            trajectory=trajectory,
        )

    def _build_user_message(
        self,
        task: TaskSpec,
        profile: DataProfile,
        runs: List[RunEntry],
        metrics: List[float],
    ) -> str:
        run_lines = []
        for r in runs:
            families = list(r.config.autogluon_kwargs.get("hyperparameters", {}).keys())
            run_lines.append(
                f"- {r.run_id}: metric={r.result.primary_metric} "
                f"families={families} rationale={r.agent_rationale[:80]}"
            )
        return (
            f"## Task\n{task.task_name} | {task.task_type} | target={task.target_column}\n"
            f"{task.description}\n\n"
            f"## Data Profile\n{profile.summary}\n\n"
            f"## Run History\n" + "\n".join(run_lines) + "\n\n"
            f"## Metric Progression\n{metrics}\n\n"
            f"Summarise this session into the JSON CaseEntry format."
        )
