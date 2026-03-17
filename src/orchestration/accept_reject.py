from __future__ import annotations
from typing import Optional
from src.models.nodes import ExperimentNode


class AcceptReject:
    """Evaluates whether a child node improves over its parent."""

    def __init__(self, higher_is_better: bool = True, min_delta: float = 0.001) -> None:
        self.higher_is_better = higher_is_better
        self.min_delta = min_delta

    def evaluate(self, parent: Optional[ExperimentNode], child: ExperimentNode) -> bool:
        # Root node: always accept if it has a valid result
        if parent is None:
            return child.has_result()

        parent_metric = parent.primary_metric()
        child_metric = child.primary_metric()

        if child_metric is None:
            return False
        if parent_metric is None:
            return True

        if self.higher_is_better:
            return child_metric >= parent_metric + self.min_delta
        else:
            return child_metric <= parent_metric - self.min_delta
