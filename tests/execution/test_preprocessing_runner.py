import pytest
import pandas as pd
from pathlib import Path
from src.models.preprocessing import PreprocessingPlan
from src.execution.preprocessing_runner import PreprocessingExecutor


def test_identity_copies_file(tmp_path):
    # Write a tiny CSV
    src = tmp_path / "data.csv"
    src.write_text("a,b,label\n1,2,0\n3,4,1\n")

    plan = PreprocessingPlan(strategy="identity")
    executor = PreprocessingExecutor()
    out_path = executor.run(str(src), plan, str(tmp_path / "out"))

    # Output file must exist and have same content
    assert Path(out_path).exists()
    original = pd.read_csv(str(src))
    result = pd.read_csv(out_path)
    pd.testing.assert_frame_equal(original, result)


def test_identity_output_named_preprocessed_data(tmp_path):
    src = tmp_path / "data.csv"
    src.write_text("a,label\n1,0\n")
    plan = PreprocessingPlan(strategy="identity")
    out_path = PreprocessingExecutor().run(str(src), plan, str(tmp_path / "out"))
    assert Path(out_path).name == "preprocessed_data.csv"


def test_unknown_strategy_raises(tmp_path):
    src = tmp_path / "data.csv"
    src.write_text("a,label\n1,0\n")
    plan = PreprocessingPlan(strategy="generated", code="def preprocess(df, t): return df")
    with pytest.raises(NotImplementedError):
        PreprocessingExecutor().run(str(src), plan, str(tmp_path / "out"))
