from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional
from src.llm.backend import LLMBackend, Message
from src.models.task import ExperimentPlan, TaskSpec
from src.models.results import DataProfile, RunEntry


class SelectorAgent:
    """
    Converts a natural language hypothesis into a concrete ExperimentPlan.
    Calls the LLM with the selector prompt, parses JSON response.
    Retries on invalid JSON up to max_retries times.
    """

    def __init__(
        self,
        llm: LLMBackend,
        prompt_path: str = "prompts/selector.md",
        max_retries: int = 3,
        temperature: float = 0.3,
    ) -> None:
        self._llm = llm
        self._system_prompt = Path(prompt_path).read_text()
        self._max_retries = max_retries
        self._temperature = temperature

    def select(
        self,
        hypothesis: str,
        task: TaskSpec,
        data_profile: DataProfile,
        prior_runs: List[RunEntry],
    ) -> ExperimentPlan:
        user_message = self._build_user_message(hypothesis, task, data_profile, prior_runs)
        messages = [
            Message(role="system", content=self._system_prompt),
            Message(role="user", content=user_message),
        ]

        last_error = None
        for attempt in range(self._max_retries):
            response = self._llm.complete(messages=messages, temperature=self._temperature)
            try:
                return ExperimentPlan.model_validate_json(response)
            except Exception as e:
                last_error = e
                # Add the failed response + retry instruction to message history
                messages.append(Message(role="assistant", content=response))
                messages.append(Message(
                    role="user",
                    content=f"Your response was not valid JSON matching the ExperimentPlan schema. "
                            f"Error: {e}. Please respond with ONLY the JSON object."
                ))

        raise ValueError(
            f"Failed to get valid ExperimentPlan after {self._max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def _build_user_message(
        self,
        hypothesis: str,
        task: TaskSpec,
        data_profile: DataProfile,
        prior_runs: List[RunEntry],
    ) -> str:
        history_text = ""
        if prior_runs:
            summaries = []
            for r in prior_runs[-5:]:  # last 5 runs
                metric = r.result.primary_metric
                summaries.append(
                    f"- Run {r.run_id}: metric={metric}, rationale={r.agent_rationale[:100]}"
                )
            history_text = "\n## Prior Runs\n" + "\n".join(summaries)

        return (
            f"## Task\n"
            f"Name: {task.task_name}\n"
            f"Type: {task.task_type}\n"
            f"Target: {task.target_column}\n"
            f"Description: {task.description}\n\n"
            f"## Data Profile\n"
            f"{data_profile.summary}\n"
            f"Rows: {data_profile.n_rows}, Features: {data_profile.n_features}\n"
            f"Class balance ratio: {data_profile.class_balance_ratio:.2f}\n"
            f"Missing rate: {data_profile.missing_rate:.2%}\n"
            f"Feature types: {data_profile.feature_types}\n"
            f"Suspected leakage columns: {data_profile.suspected_leakage_cols}\n"
            f"{history_text}\n\n"
            f"## Hypothesis to implement\n"
            f"{hypothesis}\n\n"
            f"Respond with ONLY the JSON ExperimentPlan."
        )
