import pytest
from src.models.campaign import CampaignConfig, SessionSummary, CampaignResult


def test_campaign_config_defaults():
    cfg = CampaignConfig()
    assert cfg.max_sessions == 5
    assert cfg.plateau_threshold == 0.002
    assert cfg.plateau_window == 3


def test_session_summary_requires_fields():
    s = SessionSummary(
        session_id="s1",
        best_metric=0.87,
        preprocessing_strategy="identity",
        session_dir="/tmp/s1",
        duration_seconds=42.0,
        error_message=None,
    )
    assert s.session_id == "s1"
    assert s.best_metric == 0.87


def test_session_summary_none_metric():
    # A session where all runs failed has best_metric=None
    s = SessionSummary(
        session_id="s2",
        best_metric=None,
        preprocessing_strategy="identity",
        session_dir="/tmp/s2",
        duration_seconds=5.0,
    )
    assert s.best_metric is None


def test_session_summary_error_message():
    s = SessionSummary(
        session_id="s3",
        best_metric=None,
        preprocessing_strategy="identity",
        session_dir="/tmp/s3",
        duration_seconds=1.0,
        error_message="AutoGluon raised RuntimeError",
    )
    assert s.error_message == "AutoGluon raised RuntimeError"


def test_campaign_result_serialises():
    import json
    r = CampaignResult(
        campaign_id="c1",
        task_name="titanic",
        started_at="2026-03-20T00:00:00",
        sessions=[],
        best_metric=None,
        best_session_id=None,
        stopped_reason="budget",
    )
    blob = json.loads(r.model_dump_json())
    assert blob["stopped_reason"] == "budget"
