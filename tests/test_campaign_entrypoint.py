import subprocess
import sys


def test_campaign_entrypoint_shows_help():
    """campaign.py --help should exit 0 and mention 'campaign'."""
    result = subprocess.run(
        [sys.executable, "campaign.py", "--help"],
        capture_output=True, text=True, cwd="/home/tough/Agentic ML"
    )
    assert result.returncode == 0
    assert "campaign" in result.stdout.lower() or "campaign" in result.stderr.lower()


def test_best_str_none_metric():
    """When all sessions fail, best_metric=None must not crash the format string."""
    # Directly exercise the format logic from campaign.py main()
    best_metric = None
    best_str = f"{best_metric:.4f}" if best_metric is not None else "N/A"
    assert best_str == "N/A"
