from pathlib import Path

from src.models.uplift import UpliftProjectContract, UpliftTableSchema
from src.uplift.eda import UpliftEDAAgent, render_eda_markdown, run_eda_phase
from src.uplift.llm_client import make_chat_llm


FIXTURE_DIR = Path("tests/fixtures/uplift")


def _contract(data_dir: Path = FIXTURE_DIR) -> UpliftProjectContract:
    return UpliftProjectContract(
        task_name="retailhero-uplift",
        table_schema=UpliftTableSchema(
            clients_table=str(data_dir / "clients.csv"),
            purchases_table=str(data_dir / "purchases.csv"),
            train_table=str(data_dir / "uplift_train.csv"),
            scoring_table=str(data_dir / "uplift_test.csv"),
            products_table=str(data_dir / "products.csv"),
        ),
        entity_key="client_id",
        treatment_column="treatment_flg",
        target_column="target",
    )


def test_eda_agent_profiles_dataset_without_llm():
    report = UpliftEDAAgent(_contract(), llm=None, purchases_sample_rows=10).run()

    assert report.table_rows["train"] == 8
    assert report.table_rows["scoring"] == 4
    assert report.treatment_counts == {0: 4, 1: 4}
    assert report.target_counts == {0: 4, 1: 4}
    assert report.average_treatment_effect == 0.0
    assert report.purchase_summary["sample_transactions"] == 4.0
    assert any(finding.topic == "dataset_shape" for finding in report.findings)
    assert report.drafted_hypotheses == []


def test_eda_agent_uses_stub_llm_to_draft_hypotheses(tmp_path):
    report = run_eda_phase(
        _contract(),
        make_chat_llm("stub", "stub"),
        output_dir=tmp_path,
        purchases_sample_rows=10,
    )

    assert report.llm_summary
    assert len(report.drafted_hypotheses) >= 1
    assert any(
        "recency" in " ".join(h.suggested_features).lower()
        for h in report.drafted_hypotheses
    )
    assert (tmp_path / "eda_report.json").exists()
    assert (tmp_path / "eda_report.md").exists()
    assert "EDA Agent Report" in render_eda_markdown(report)
