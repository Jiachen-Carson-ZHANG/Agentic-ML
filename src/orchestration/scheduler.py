from __future__ import annotations
from typing import List, Optional
from src.models.nodes import ExperimentNode


class Scheduler:
    """Controls session budget, stage transitions, and warm-up tracking."""

    def __init__(
        self,
        num_candidates: int = 3,
        min_warmup_runs: int = 1,
        max_optimize_iterations: int = 5,
        plateau_threshold: float = 0.001,
        plateau_patience: int = 3,
    ) -> None:
        self.num_candidates = num_candidates
        self.min_warmup_runs = min_warmup_runs
        self.max_optimize_iterations = max_optimize_iterations
        self.plateau_threshold = plateau_threshold
        self.plateau_patience = plateau_patience
        self.stage: str = "warmup"
        self._warmup_nodes: List[ExperimentNode] = []
        self._optimize_count: int = 0

    def record_warmup_run(self, node: ExperimentNode) -> None:
        self._warmup_nodes.append(node)

    def record_optimize_run(self) -> None:
        self._optimize_count += 1

    def should_advance_to_optimization(self) -> bool:
        valid = [n for n in self._warmup_nodes if n.has_result()]
        return len(valid) >= self.num_candidates

    def advance_to_optimization(self) -> None:
        self.stage = "optimize"

    def should_stop(self) -> bool:
        return self._optimize_count >= self.max_optimize_iterations

    def get_stage(self) -> str:
        return self.stage
