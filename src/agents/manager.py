from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from src.llm.backend import LLMBackend
from src.models.nodes import SearchContext


class ActionType(str, Enum):
    SELECT = "select"        # warmup: turn hypothesis into ExperimentPlan
    REFINE = "refine"        # optimize: propose config change from results
    DEBUG = "debug"          # fix a failed run
    ACCEPT = "accept"        # promote child to incumbent
    REJECT = "reject"        # revert to parent
    STOP = "stop"            # session complete


@dataclass
class Action:
    action_type: ActionType
    reason: str = ""


class ExperimentManager:
    """
    Top-level agent orchestrator. Decides the next action based on SearchContext.
    Routes to SelectorAgent (warmup) or RefinerAgent (optimize) as appropriate.
    """

    def __init__(self, llm: LLMBackend) -> None:
        self._llm = llm

    def next_action(self, context: SearchContext) -> Action:
        # Budget exhausted → stop
        if context.budget_remaining <= 0:
            return Action(ActionType.STOP, reason="Budget exhausted")

        # Current node failed → debug
        if (context.current_node.status.value == "failed"
                and context.current_node.debug_depth < 2):
            return Action(ActionType.DEBUG, reason="Current node failed, attempting debug")

        # Warmup phase → select next candidate config
        if context.stage == "warmup":
            return Action(ActionType.SELECT, reason="Warmup: selecting experiment config for candidate")

        # Optimize phase → refine incumbent
        if context.stage == "optimize":
            if context.incumbent is not None:
                return Action(ActionType.REFINE, reason="Optimize: refining incumbent config")
            return Action(ActionType.SELECT, reason="Optimize: no incumbent yet, selecting baseline")

        return Action(ActionType.STOP, reason="Unknown stage")
