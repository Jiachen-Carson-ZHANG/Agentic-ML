"""
execution_agents.py
===================
Compatibility shim — re-exports every public symbol from the three phase files.

The pipeline has been split into:
  phase3_execution.py  — CodeConfigAgent, TrainingExecutionAgent, TrainingResult
  phase6_memory.py     — ExperimentMemoryAgent, RetryControllerAgent, RetryDecision
  phase7_output.py     — ManualBenchmarkAgent, ReportingAgent, AutoLiftOrchestrator,
                         BenchmarkResult

Any existing notebook cell that does:
    from execution_agents import TrainingExecutionAgent
will continue to work unchanged.
"""

from phase3_execution import (
    VALID_LEARNER_FAMILIES,
    VALID_BASE_ESTIMATORS,
    TrainingResult,
    CodeConfigAgent,
    TrainingExecutionAgent,
)

from phase6_memory import (
    RetryDecision,
    ExperimentMemoryAgent,
    RetryControllerAgent,
    _params_hash,
)

from phase7_output import (
    BenchmarkResult,
    ManualBenchmarkAgent,
    ReportingAgent,
    AutoLiftOrchestrator,
    _MANUAL_BASELINE_CONFIG,
    _MANUAL_BASELINE_FEATURES,
)

__all__ = [
    # Phase III
    "VALID_LEARNER_FAMILIES",
    "VALID_BASE_ESTIMATORS",
    "TrainingResult",
    "CodeConfigAgent",
    "TrainingExecutionAgent",
    # Phase VI
    "RetryDecision",
    "ExperimentMemoryAgent",
    "RetryControllerAgent",
    "_params_hash",
    # Phase VII
    "BenchmarkResult",
    "ManualBenchmarkAgent",
    "ReportingAgent",
    "AutoLiftOrchestrator",
    "_MANUAL_BASELINE_CONFIG",
    "_MANUAL_BASELINE_FEATURES",
]
