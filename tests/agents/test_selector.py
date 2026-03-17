import pytest
import json
from unittest.mock import MagicMock
from src.agents.selector import SelectorAgent
from src.models.task import ExperimentPlan, TaskSpec
from src.models.results import DataProfile
from src.llm.backend import LLMBackend, Message


def make_task() -> TaskSpec:
    return TaskSpec(
        task_name="titanic", task_type="binary",
        data_path="data/titanic_train.csv", target_column="Survived",
        eval_metric="roc_auc", description="Predict survival"
    )


def make_profile() -> DataProfile:
    return DataProfile(
        n_rows=891, n_features=11,
        feature_types={"numeric": 7, "categorical": 4},
        class_balance_ratio=0.61,
        missing_rate=0.02,
        summary="891 rows, binary target, mild class imbalance"
    )


def make_valid_plan_json() -> str:
    return json.dumps({
        "eval_metric": "roc_auc",
        "model_families": ["GBM", "XGB"],
        "presets": "medium_quality",
        "time_limit": 120,
        "feature_policy": {"exclude_columns": [], "include_columns": []},
        "validation_policy": {"holdout_frac": 0.2, "num_bag_folds": 0},
        "hyperparameters": None,
        "use_fit_extra": False,
        "rationale": "GBM and XGB are strong baselines for tabular binary classification."
    })


def test_selector_returns_experiment_plan():
    mock_backend = MagicMock(spec=LLMBackend)
    mock_backend.complete.return_value = make_valid_plan_json()

    agent = SelectorAgent(llm=mock_backend, prompt_path="prompts/selector.md")
    plan = agent.select(
        hypothesis="Try gradient boosting as a baseline",
        task=make_task(),
        data_profile=make_profile(),
        prior_runs=[]
    )

    assert isinstance(plan, ExperimentPlan)
    assert plan.eval_metric == "roc_auc"
    assert "GBM" in plan.model_families


def test_selector_calls_llm_once():
    mock_backend = MagicMock(spec=LLMBackend)
    mock_backend.complete.return_value = make_valid_plan_json()

    agent = SelectorAgent(llm=mock_backend, prompt_path="prompts/selector.md")
    agent.select(
        hypothesis="Try GBM baseline",
        task=make_task(),
        data_profile=make_profile(),
        prior_runs=[]
    )
    mock_backend.complete.assert_called_once()


def test_selector_includes_hypothesis_in_prompt():
    mock_backend = MagicMock(spec=LLMBackend)
    mock_backend.complete.return_value = make_valid_plan_json()

    agent = SelectorAgent(llm=mock_backend, prompt_path="prompts/selector.md")
    agent.select(
        hypothesis="Try random forest for robustness",
        task=make_task(),
        data_profile=make_profile(),
        prior_runs=[]
    )
    call_args = mock_backend.complete.call_args
    messages = call_args.kwargs.get("messages") or call_args.args[0]
    user_content = next(m.content for m in messages if m.role == "user")
    assert "random forest" in user_content.lower()


def test_selector_retries_on_invalid_json():
    mock_backend = MagicMock(spec=LLMBackend)
    mock_backend.complete.side_effect = [
        "this is not json",
        make_valid_plan_json()
    ]

    agent = SelectorAgent(llm=mock_backend, prompt_path="prompts/selector.md")
    plan = agent.select(
        hypothesis="Try GBM",
        task=make_task(),
        data_profile=make_profile(),
        prior_runs=[]
    )
    assert isinstance(plan, ExperimentPlan)
    assert mock_backend.complete.call_count == 2


def test_selector_raises_after_max_retries():
    mock_backend = MagicMock(spec=LLMBackend)
    mock_backend.complete.return_value = "not json at all"

    agent = SelectorAgent(llm=mock_backend, prompt_path="prompts/selector.md", max_retries=2)
    with pytest.raises(ValueError, match="Failed to get valid ExperimentPlan"):
        agent.select(
            hypothesis="Try GBM",
            task=make_task(),
            data_profile=make_profile(),
            prior_runs=[]
        )
