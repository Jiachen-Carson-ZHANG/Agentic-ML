# Schema Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove duplicate fields, dead fields, and a confusing class name from the core data model so the schema is clean, non-redundant, and self-explanatory.

**Architecture:** Three schema changes cascade through ~10 files: (1) `RunResult` loses fields that belong elsewhere or are never used, (2) `RunConfig` loses IDs that are owned by the outer record, (3) `RunEntry` is renamed `ExperimentRun` and loses `agent_review` (ReviewerAgent not built). Each task is: update the schema + update tests, then update callers.

**Tech Stack:** Python 3.12, Pydantic v2, pytest

---

## What changes and why

| Class | Remove | Reason |
|-------|--------|--------|
| `RunResult` | `run_id` | Owned by `ExperimentRun`, duplicated here and in `RunConfig` |
| `RunResult` | `artifacts_dir` | Same path as `RunConfig.output_dir`, duplicated |
| `RunResult` | `raw_info` | Always `{}`, never read |
| `RunResult` | `diagnostics_overfitting_gap` | Diagnostic concern — belongs in `RunDiagnostics` only |
| `RunConfig` | `run_id`, `node_id` | Execution-time IDs — owned by `ExperimentRun` |
| `RunDiagnostics` | `data_profile_ref` | Always `""`, never read |
| `RunDiagnostics` | `feature_importances` | Always `{}`, never populated |
| `RunDiagnostics` | `change_description` | Always `""`, never populated |
| `RunEntry` | rename → `ExperimentRun` | "Entry" implies a log line; this is the full composite record of a run |
| `RunEntry` | `agent_review` | ReviewerAgent not built yet, always `""` |

**Overfitting gap flow after cleanup:**
`ResultParser.from_predictor` computes it and returns `(RunResult, Optional[float])`.
`AutoGluonRunner.run` returns `(RunResult, Optional[float])`.
`session.execute_node` puts the float into `RunDiagnostics.overfitting_gap`.

---

## Task 1: Update `RunResult` schema and its tests

**Files:**
- Modify: `src/models/results.py`
- Modify: `tests/models/test_models.py`

**Step 1: Update the test to match the target schema**

In `tests/models/test_models.py`, replace `test_run_result_failed`:

```python
def test_run_result_failed():
    result = RunResult(
        status="failed",
        primary_metric=None,
        leaderboard=[],
        best_model_name=None,
        fit_time_seconds=0.0,
        error="AutoGluon crashed: OOM",
    )
    assert result.status == "failed"
    assert result.primary_metric is None
```

**Step 2: Run test to confirm it fails**

```bash
cd "/home/tough/Agentic ML"
pytest tests/models/test_models.py::test_run_result_failed -v
```

Expected: FAIL — `RunResult` still requires `run_id`.

**Step 3: Update `RunResult` in `src/models/results.py`**

Replace the `RunResult` class:

```python
class RunResult(BaseModel):
    """What AutoGluon returned from a fit() call. No IDs — those live in ExperimentRun."""
    status: str  # "success" | "failed"
    primary_metric: Optional[float] = None
    leaderboard: List[ModelEntry] = Field(default_factory=list)
    best_model_name: Optional[str] = None
    fit_time_seconds: float = 0.0
    error: Optional[str] = None
```

**Step 4: Run the test**

```bash
pytest tests/models/test_models.py::test_run_result_failed -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/models/results.py tests/models/test_models.py
git commit -m "refactor: remove dead fields from RunResult (run_id, artifacts_dir, raw_info, diagnostics_overfitting_gap)"
```

---

## Task 2: Update `RunDiagnostics` schema and its tests

**Files:**
- Modify: `src/models/results.py`
- Modify: `tests/models/test_models.py`
- Modify: `tests/memory/test_run_store.py`

**Step 1: Update `RunDiagnostics` in `src/models/results.py`**

Replace the `RunDiagnostics` class:

```python
class RunDiagnostics(BaseModel):
    """Computed observations after a run completes. All fields optional — never required for correctness."""
    overfitting_gap: Optional[float] = None
    metric_vs_parent: Optional[float] = None
    failure_mode: Optional[str] = None
```

**Step 2: Run existing tests to see what breaks**

```bash
pytest tests/ -v 2>&1 | grep -E "FAILED|ERROR" | head -20
```

**Step 3: Fix any test that constructs `RunDiagnostics` with removed fields**

In `tests/memory/test_run_store.py`, the `make_run_entry` helper constructs `RunDiagnostics(metric_vs_parent=0.05)` — that field still exists, so it's fine. But check for any use of `data_profile_ref`, `feature_importances`, or `change_description` in tests:

```bash
grep -r "data_profile_ref\|feature_importances\|change_description" tests/
```

Remove any such usages found.

**Step 4: Run tests**

```bash
pytest tests/ -v 2>&1 | grep -E "FAILED|ERROR"
```

Expected: only failures from callers not yet updated (session.py sets `data_profile_ref`).

**Step 5: Commit**

```bash
git add src/models/results.py
git commit -m "refactor: remove dead fields from RunDiagnostics (data_profile_ref, feature_importances, change_description)"
```

---

## Task 3: Update `RunConfig` schema and `ConfigMapper`

**Files:**
- Modify: `src/models/task.py`
- Modify: `src/execution/config_mapper.py`
- Modify: `tests/models/test_models.py`
- Modify: `tests/execution/test_execution.py`

**Step 1: Update `RunConfig` in `src/models/task.py`**

Replace `RunConfig`:

```python
class RunConfig(BaseModel):
    """AutoGluon-ready execution config. IDs (run_id, node_id) live in ExperimentRun."""
    autogluon_kwargs: Dict[str, Any]
    data_path: str
    output_dir: str
```

**Step 2: Update `ConfigMapper.to_run_config` in `src/execution/config_mapper.py`**

Remove `run_id` and `node_id` from the signature and the returned `RunConfig`:

```python
@staticmethod
def to_run_config(
    plan: ExperimentPlan,
    data_path: str,
    output_dir: str,
) -> RunConfig:
    kwargs: dict = {
        "eval_metric": plan.eval_metric,
        "time_limit": plan.time_limit,
        "presets": plan.presets,
    }
    if plan.model_families:
        kwargs["hyperparameters"] = {family: {} for family in plan.model_families}
    val = plan.validation_policy
    if val.get("num_bag_folds", 0) > 0:
        kwargs["num_bag_folds"] = val["num_bag_folds"]
    elif val.get("holdout_frac", 0.0) > 0.0:
        kwargs["holdout_frac"] = val["holdout_frac"]
    excluded = plan.feature_policy.get("exclude_columns", [])
    if excluded:
        kwargs["excluded_columns"] = excluded
    if plan.hyperparameters:
        kwargs["hyperparameters"] = plan.hyperparameters
    return RunConfig(
        autogluon_kwargs=kwargs,
        data_path=data_path,
        output_dir=output_dir,
    )
```

**Step 3: Update `test_run_config_has_node_id` in `tests/models/test_models.py`**

Rename and update the test — `RunConfig` no longer has `node_id`:

```python
def test_run_config_has_output_dir():
    config = RunConfig(
        autogluon_kwargs={"eval_metric": "roc_auc", "time_limit": 120},
        data_path="data/test.csv",
        output_dir="experiments/test/runs/run_0001",
    )
    assert config.output_dir == "experiments/test/runs/run_0001"
```

**Step 4: Update `make_run_entry` helper in `tests/memory/test_run_store.py`**

```python
config = RunConfig(
    autogluon_kwargs={"eval_metric": "roc_auc"},
    data_path="data/test.csv",
    output_dir=f"experiments/test/runs/{run_id}",
)
```

**Step 5: Check for other test files constructing `RunConfig` with `run_id`/`node_id`**

```bash
grep -r "RunConfig(" tests/
```

Fix any found.

**Step 6: Run tests**

```bash
pytest tests/ -v 2>&1 | grep -E "FAILED|ERROR"
```

**Step 7: Commit**

```bash
git add src/models/task.py src/execution/config_mapper.py tests/
git commit -m "refactor: remove run_id and node_id from RunConfig — IDs owned by ExperimentRun"
```

---

## Task 4: Rename `RunEntry` → `ExperimentRun`, remove `agent_review`

**Files:**
- Modify: `src/models/results.py`
- Modify: all files that import `RunEntry` (run_store, session, nodes, distiller, context_builder, refiner, selector, ideator)
- Modify: all test files that import `RunEntry`

**Step 1: Rename in `src/models/results.py`**

Replace the `RunEntry` class:

```python
class ExperimentRun(BaseModel):
    """Complete record of one experiment run — the unit stored in decisions.jsonl.

    Composite of: what the agent planned (plan), how it was translated to AutoGluon kwargs (config),
    what AutoGluon returned (result), and what we computed afterward (diagnostics).
    """
    run_id: str
    node_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    config: RunConfig
    result: RunResult
    diagnostics: RunDiagnostics = Field(default_factory=RunDiagnostics)
    plan: Optional[ExperimentPlan] = None
    agent_rationale: str = ""
```

**Step 2: Update all src imports — replace `RunEntry` with `ExperimentRun`**

Files to update (replace every `RunEntry` reference):

```bash
# Check all occurrences
grep -rn "RunEntry" src/
```

Files: `src/memory/run_store.py`, `src/session.py`, `src/models/nodes.py`, `src/memory/distiller.py`, `src/memory/context_builder.py`, `src/agents/refiner.py`, `src/agents/selector.py`

In each file:
- Change `from src.models.results import ..., RunEntry` → `..., ExperimentRun`
- Replace all `RunEntry` type hints and usages with `ExperimentRun`

**Step 3: Update all test imports**

```bash
grep -rn "RunEntry" tests/
```

In each test file found, replace `RunEntry` with `ExperimentRun` in imports and usages.

Specifically in `tests/memory/test_run_store.py`, update `make_run_entry` return type:

```python
def make_run_entry(run_id: str, metric: float, status: str = "success") -> ExperimentRun:
    ...
    return ExperimentRun(
        run_id=run_id, node_id="node_001",
        timestamp=datetime(2026, 3, 16, 12, 0, 0),
        config=config, result=result,
        diagnostics=RunDiagnostics(metric_vs_parent=0.05),
        agent_rationale="test rationale",
    )
```

**Step 4: Run tests**

```bash
pytest tests/ -v 2>&1 | grep -E "FAILED|ERROR"
```

**Step 5: Commit**

```bash
git add src/ tests/
git commit -m "refactor: rename RunEntry → ExperimentRun, remove agent_review (ReviewerAgent not yet built)"
```

---

## Task 5: Update `ResultParser` and `AutoGluonRunner` for new return type

`ResultParser.from_predictor` now returns `(RunResult, Optional[float])` where the float is `overfitting_gap`. `AutoGluonRunner.run` propagates the same tuple.

**Files:**
- Modify: `src/execution/result_parser.py`
- Modify: `src/execution/autogluon_runner.py`
- Modify: `src/session.py`
- Modify: `tests/test_result_parser.py`

**Step 1: Update `ResultParser` in `src/execution/result_parser.py`**

```python
from typing import Any, Optional, Tuple

class ResultParser:
    @staticmethod
    def from_predictor(
        predictor: Any,
        fit_time: float,
        primary_metric_value: float,
    ) -> Tuple[RunResult, Optional[float]]:
        leaderboard_entries = []
        overfitting_gap = None
        try:
            lb = predictor.leaderboard(extra_info=True)
            best_row = lb.iloc[0]
            score_train = float(best_row["score_train"]) if "score_train" in lb.columns else None
            score_val = float(best_row["score_val"])
            if score_train is not None:
                overfitting_gap = round(score_train - score_val, 4)
            for row in lb.itertuples():
                leaderboard_entries.append(ModelEntry(
                    model_name=row.model,
                    score_val=row.score_val,
                    fit_time=row.fit_time,
                    pred_time=row.pred_time,
                    stack_level=getattr(row, "stack_level", 1),
                    score_train=getattr(row, "score_train", None),
                ))
        except Exception as e:
            logger.warning("leaderboard(extra_info=True) failed, falling back: %s", e)
            try:
                lb = predictor.leaderboard()
                for row in lb.itertuples():
                    leaderboard_entries.append(ModelEntry(
                        model_name=row.model,
                        score_val=row.score_val,
                        fit_time=row.fit_time,
                        pred_time=row.pred_time,
                        stack_level=getattr(row, "stack_level", 1),
                    ))
            except Exception:
                pass

        result = RunResult(
            status="success",
            primary_metric=primary_metric_value,
            leaderboard=leaderboard_entries,
            best_model_name=getattr(predictor, "model_best", None),
            fit_time_seconds=fit_time,
        )
        return result, overfitting_gap

    @staticmethod
    def from_error(error_msg: str) -> RunResult:
        return RunResult(
            status="failed",
            error=error_msg,
        )
```

**Step 2: Update `AutoGluonRunner.run` in `src/execution/autogluon_runner.py`**

Change return type to `Tuple[RunResult, Optional[float]]`:

```python
from typing import Optional, Tuple
from src.models.results import RunResult

def run(self, config: RunConfig) -> Tuple[RunResult, Optional[float]]:
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        return ResultParser.from_error(
            "AutoGluon not installed. Run: pip install autogluon.tabular"
        ), None

    os.makedirs(config.output_dir, exist_ok=True)
    df = pd.read_csv(config.data_path)
    log_path = os.path.join(config.output_dir, "training.log")

    kwargs = dict(config.autogluon_kwargs)
    kwargs["path"] = config.output_dir

    predictor = TabularPredictor(
        label=self.target_column,
        verbosity=1,
        **{k: v for k, v in kwargs.items()
           if k in ("eval_metric", "path", "problem_type")}
    )

    fit_kwargs = {k: v for k, v in kwargs.items()
                  if k not in ("eval_metric", "path", "problem_type")}

    start = time.time()
    try:
        with _log_to_file(log_path):
            predictor.fit(df, **fit_kwargs)
    except Exception as e:
        return ResultParser.from_error(str(e)), None
    fit_time = time.time() - start

    lb = predictor.leaderboard(silent=True)
    primary_metric = float(lb["score_val"].max()) if not lb.empty else 0.0

    return ResultParser.from_predictor(
        predictor=predictor,
        fit_time=fit_time,
        primary_metric_value=primary_metric,
    )
```

**Step 3: Update `tests/test_result_parser.py`**

```python
def test_overfitting_gap_computed():
    predictor = _make_predictor(
        val_scores=[0.87, 0.85],
        train_scores=[0.95, 0.93],
    )
    result, overfitting_gap = ResultParser.from_predictor(predictor, 10.0, 0.87)
    assert result.leaderboard[0].score_train == pytest.approx(0.95)
    assert overfitting_gap == pytest.approx(0.95 - 0.87)


def test_overfitting_gap_none_when_extra_info_fails():
    lb_basic = pd.DataFrame({
        "model": ["WeightedEnsemble_L2"],
        "score_val": [0.87],
        "fit_time": [10.0],
        "pred_time": [0.1],
        "stack_level": [2],
    })
    p = MagicMock()
    p.model_best = "WeightedEnsemble_L2"
    def leaderboard_side_effect(extra_info=False):
        if extra_info:
            raise ValueError("extra_info not supported")
        return lb_basic
    p.leaderboard.side_effect = leaderboard_side_effect
    result, overfitting_gap = ResultParser.from_predictor(p, 10.0, 0.87)
    assert overfitting_gap is None
    assert len(result.leaderboard) == 1
    assert result.leaderboard[0].score_train is None
```

**Step 4: Update `session.execute_node` in `src/session.py`**

Change the `execute_node` method to unpack the tuple from runner and update ConfigMapper call:

```python
def execute_node(self, node: ExperimentNode, data_profile: DataProfile) -> ExperimentRun:
    self._run_counter += 1
    run_id = f"run_{self._run_counter:04d}"
    run_dir = str(self._session_dir / "runs" / run_id)

    config = ConfigMapper.to_run_config(
        plan=node.plan,
        data_path=self.task.data_path,
        output_dir=run_dir,
    )

    # Write per-run config snapshot
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    run_config_snapshot = {
        "run_id": run_id,
        "node_id": node.node_id,
        "autogluon_kwargs": config.autogluon_kwargs,
        "experiment_plan": node.plan.model_dump(),
    }
    (Path(run_dir) / "run_config.json").write_text(
        json.dumps(run_config_snapshot, indent=2, default=str)
    )

    self.tree.update_node(node.model_copy(update={
        "status": NodeStatus.RUNNING,
        "config": config,
    }))

    result, overfitting_gap = self._runner.run(config)

    diagnostics = RunDiagnostics(
        failure_mode="execution_error" if result.status == "failed" else None,
    )

    entry = ExperimentRun(
        run_id=run_id,
        node_id=node.node_id,
        config=config,
        result=result,
        diagnostics=diagnostics,
        agent_rationale=node.plan.rationale,
        plan=node.plan,
    )

    if result.status == "success":
        parent_node = self.tree.get_node(node.parent_id) if node.parent_id else None
        parent_metric = parent_node.primary_metric() if parent_node and parent_node.has_result() else None
        metric_vs_parent = None
        if parent_metric is not None and result.primary_metric is not None:
            metric_vs_parent = round(result.primary_metric - parent_metric, 4)
        entry.diagnostics = RunDiagnostics(
            overfitting_gap=overfitting_gap,
            metric_vs_parent=metric_vs_parent,
        )

    self.run_store.append(entry)
    new_status = NodeStatus.SUCCESS if result.status == "success" else NodeStatus.FAILED
    self.tree.update_node(node.model_copy(update={
        "status": new_status,
        "config": config,
        "entry": entry,
    }))
    return entry
```

Also update the `session.py` import line:
```python
from src.models.results import DataProfile, ExperimentRun, RunResult, RunDiagnostics
```

And update return type annotations in `run()` method from `RunEntry` to `ExperimentRun`.

**Step 5: Run all tests**

```bash
pytest tests/ -v
```

Expected: all pass.

**Step 6: Commit**

```bash
git add src/execution/result_parser.py src/execution/autogluon_runner.py src/session.py tests/test_result_parser.py
git commit -m "refactor: ResultParser returns (RunResult, overfitting_gap) tuple — removes diagnostics_overfitting_gap from RunResult"
```

---

## Task 6: Full test suite + docs update

**Step 1: Run full test suite**

```bash
cd "/home/tough/Agentic ML"
pytest tests/ -v
```

Expected: all tests pass. If any fail, fix them before proceeding.

**Step 2: Quick smoke check — confirm imports work**

```bash
python3 -c "
from src.models.results import ExperimentRun, RunResult, RunDiagnostics, ModelEntry, DataProfile
from src.models.task import TaskSpec, ExperimentPlan, RunConfig
from src.models.nodes import ExperimentNode, CaseEntry, SearchContext
from src.execution.result_parser import ResultParser
from src.execution.config_mapper import ConfigMapper
print('All imports OK')
"
```

**Step 3: Update `docs/changes/implementation-log.md`**

Append:

```markdown
## 2026-03-20 — Schema cleanup: RunResult, RunDiagnostics, RunConfig, RunEntry→ExperimentRun

**What changed:**
- `RunResult` stripped to execution output only: removed `run_id`, `artifacts_dir`, `raw_info`, `diagnostics_overfitting_gap`
- `RunDiagnostics` stripped to computed observations only: removed `data_profile_ref`, `feature_importances`, `change_description`
- `RunConfig` stripped to AutoGluon kwargs wrapper: removed `run_id`, `node_id` (owned by `ExperimentRun`)
- `RunEntry` renamed to `ExperimentRun`; removed `agent_review` (ReviewerAgent not built yet)
- `ResultParser.from_predictor` now returns `(RunResult, Optional[float])` — overfitting_gap surfaced as a separate value
- `AutoGluonRunner.run` propagates the tuple; `session.execute_node` unpacks and puts gap into `RunDiagnostics`

**Why:** Fields were duplicated across 2-3 classes (run_id in RunEntry + RunResult + RunConfig), dead fields accumulated (raw_info, data_profile_ref), and RunEntry's name implied a log line rather than a composite record.
```

**Step 4: Commit docs**

```bash
git add docs/changes/implementation-log.md
git commit -m "docs: log schema cleanup changes"
```
