import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.models.campaign import CampaignConfig
from src.models.task import TaskSpec
from src.llm.backend import LLMBackend
from src.orchestration.campaign import CampaignOrchestrator


def make_task(tmp_path) -> TaskSpec:
    csv = tmp_path / "data.csv"
    csv.write_text("f1,f2,label\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n")
    return TaskSpec(
        task_name="test", task_type="binary",
        data_path=str(csv), target_column="label",
        eval_metric="roc_auc", description="Test",
    )


def test_plateau_detection():
    cfg = CampaignConfig(plateau_threshold=0.002, plateau_window=3)
    orch = CampaignOrchestrator.__new__(CampaignOrchestrator)
    orch._config = cfg

    # Not enough sessions yet
    assert orch._is_plateau([0.87, 0.871]) is False

    # Flat — all within 0.002
    assert orch._is_plateau([0.87, 0.871, 0.870]) is True

    # Not flat — big jump in last window
    assert orch._is_plateau([0.85, 0.86, 0.871]) is False


def test_campaign_stops_at_budget(tmp_path):
    mock_llm = MagicMock(spec=LLMBackend)
    task = make_task(tmp_path)
    cfg = CampaignConfig(max_sessions=2, plateau_window=5)  # plateau never triggers

    session_metrics = [0.87, 0.88]  # two sessions

    with patch("src.orchestration.campaign.ExperimentSession") as MockSession, \
         patch("src.orchestration.campaign.PreprocessingExecutor"):
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            m = MagicMock()
            m._session_dir = tmp_path / f"sess_{call_count}"
            (tmp_path / f"sess_{call_count}").mkdir(exist_ok=True)
            inc = MagicMock()
            inc.result.primary_metric = session_metrics[call_count] if call_count < len(session_metrics) else None
            m.run_store.get_incumbent.return_value = inc
            m.run.return_value = None
            call_count += 1
            return m
        MockSession.side_effect = side_effect

        orch = CampaignOrchestrator(
            task=task, llm=mock_llm, config=cfg,
            experiments_dir=str(tmp_path / "experiments"),
        )
        result = orch.run()

    assert result.stopped_reason == "budget"
    assert len(result.sessions) == 2


def test_campaign_stops_on_plateau(tmp_path):
    mock_llm = MagicMock(spec=LLMBackend)
    task = make_task(tmp_path)
    cfg = CampaignConfig(max_sessions=10, plateau_window=3, plateau_threshold=0.002)

    call_count = 0
    flat_metrics = [0.87, 0.870, 0.871]  # 3 sessions, all within 0.002

    with patch("src.orchestration.campaign.ExperimentSession") as MockSession, \
         patch("src.orchestration.campaign.PreprocessingExecutor"):
        def side_effect(*args, **kwargs):
            nonlocal call_count
            m = MagicMock()
            m._session_dir = tmp_path / f"s_{call_count}"
            (tmp_path / f"s_{call_count}").mkdir(exist_ok=True)
            inc = MagicMock()
            inc.result.primary_metric = flat_metrics[call_count]
            m.run_store.get_incumbent.return_value = inc
            m.run.return_value = None
            call_count += 1
            return m
        MockSession.side_effect = side_effect

        orch = CampaignOrchestrator(
            task=task, llm=mock_llm, config=cfg,
            experiments_dir=str(tmp_path / "experiments"),
        )
        result = orch.run()

    assert result.stopped_reason == "plateau"
    assert len(result.sessions) == 3


def test_lower_is_better_plateau(tmp_path):
    """_is_plateau uses spread (max-min), which works for both higher and lower is better."""
    cfg = CampaignConfig(plateau_threshold=0.002, plateau_window=3)
    orch = CampaignOrchestrator.__new__(CampaignOrchestrator)
    orch._config = cfg
    # For RMSE: values decreasing (improving). Spread < threshold → plateau.
    assert orch._is_plateau([0.300, 0.301, 0.300]) is True
    # Large improvement → not plateau
    assert orch._is_plateau([0.300, 0.280, 0.260]) is False


def test_best_metric_lower_is_better(tmp_path):
    orch = CampaignOrchestrator.__new__(CampaignOrchestrator)
    orch._higher_is_better = False
    assert orch._best_metric([0.30, 0.28, 0.25]) == 0.25  # lower is better


def test_session_error_campaign_continues(tmp_path):
    """If a session throws, the campaign records the error and continues."""
    mock_llm = MagicMock(spec=LLMBackend)
    task = make_task(tmp_path)
    cfg = CampaignConfig(max_sessions=2, plateau_window=5)
    call_count = 0

    with patch("src.orchestration.campaign.ExperimentSession") as MockSession, \
         patch("src.orchestration.campaign.PreprocessingExecutor"):
        def side_effect(*args, **kwargs):
            nonlocal call_count
            m = MagicMock()
            m._session_dir = tmp_path / f"s_{call_count}"
            (tmp_path / f"s_{call_count}").mkdir(exist_ok=True)
            if call_count == 0:
                m.run.side_effect = RuntimeError("AutoGluon crashed")
            else:
                inc = MagicMock()
                inc.result.primary_metric = 0.87
                m.run_store.get_incumbent.return_value = inc
                m.run.return_value = None
            call_count += 1
            return m
        MockSession.side_effect = side_effect

        orch = CampaignOrchestrator(
            task=task, llm=mock_llm, config=cfg,
            experiments_dir=str(tmp_path / "experiments"),
        )
        result = orch.run()

    assert len(result.sessions) == 2
    assert result.sessions[0].error_message is not None
    assert result.sessions[1].best_metric == 0.87
    assert result.stopped_reason == "budget"


def test_campaign_json_written_after_each_session(tmp_path):
    """campaign.json must be written after session 1, before session 2 starts (crash-survival guarantee)."""
    mock_llm = MagicMock(spec=LLMBackend)
    task = make_task(tmp_path)
    cfg = CampaignConfig(max_sessions=2, plateau_window=5)
    call_count = 0
    campaign_json_existed_before_session2 = [False]

    with patch("src.orchestration.campaign.ExperimentSession") as MockSession, \
         patch("src.orchestration.campaign.PreprocessingExecutor"):
        def side_effect(*args, **kwargs):
            nonlocal call_count
            m = MagicMock()
            m._session_dir = tmp_path / f"s_{call_count}"
            (tmp_path / f"s_{call_count}").mkdir(exist_ok=True)
            inc = MagicMock()
            inc.result.primary_metric = 0.87
            m.run_store.get_incumbent.return_value = inc
            m.run.return_value = None
            if call_count == 1:
                # Session 1 write should have happened before session 2 is constructed
                campaigns = list((tmp_path / "experiments" / "campaigns").iterdir())
                if campaigns and (campaigns[0] / "campaign.json").exists():
                    campaign_json_existed_before_session2[0] = True
            call_count += 1
            return m
        MockSession.side_effect = side_effect

        orch = CampaignOrchestrator(
            task=task, llm=mock_llm, config=cfg,
            experiments_dir=str(tmp_path / "experiments"),
        )
        orch.run()

    assert campaign_json_existed_before_session2[0], \
        "campaign.json was not written after session 1 (crash-survival guarantee broken)"
