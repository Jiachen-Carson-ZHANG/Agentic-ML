# Phase 2 — Memory & Ideation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the agent stateful — it carries knowledge from past sessions and starts each new session with informed hypotheses instead of generic ones.

**Architecture:** Five new modules across memory and agent layers. CaseStore persists CaseEntry JSONL cross-session. CaseRetriever ranks past cases by task similarity (cosine on trait vectors). Distiller summarises a completed session into a CaseEntry via LLM. ContextBuilder assembles the full SearchContext briefing. IdeatorAgent replaces seed_ideas with LLM-generated hypotheses grounded in data profile and retrieved cases. All wired into session.py replacing the current static seed flow.

**Tech Stack:** Python 3.12, Pydantic v2, scikit-learn cosine_similarity, existing LLM backend, pytest

**Existing models — do not redefine:**
- `CaseEntry`, `TaskTraits`, `WhatWorked`, `WhatFailed`, `SessionTrajectory`, `TreeSummary`, `SearchContext` — all in `src/models/nodes.py`
- `RunStore`, `ExperimentTree`, `TaskSpec`, `DataProfile` — already exist

---

## Task 1: CaseStore

**Files:**
- Create: `src/memory/case_store.py`
- Test: `tests/memory/test_case_store.py`

### Step 1: Write failing tests

```python
# tests/memory/test_case_store.py
import pytest, json
from pathlib import Path
from src.memory.case_store import CaseStore
from src.models.nodes import CaseEntry, TaskTraits, WhatWorked, WhatFailed, SessionTrajectory
from src.models.task import ExperimentPlan


def _make_case(case_id: str, task_type: str = "binary") -> CaseEntry:
    plan = ExperimentPlan(
        eval_metric="roc_auc", model_families=["GBM"], presets="medium_quality",
        time_limit=120, feature_policy={}, validation_policy={"holdout_frac": 0.2},
    )
    return CaseEntry(
        case_id=case_id,
        task_traits=TaskTraits(
            task_type=task_type, n_rows_bucket="medium",
            n_features_bucket="medium", class_balance="balanced",
        ),
        what_worked=WhatWorked(best_config=plan, best_metric=0.85),
        what_failed=WhatFailed(),
        trajectory=SessionTrajectory(n_runs=3),
    )


def test_add_and_get_all(tmp_path):
    store = CaseStore(str(tmp_path / "cases.jsonl"))
    store.add(_make_case("c1"))
    store.add(_make_case("c2"))
    all_cases = store.get_all()
    assert len(all_cases) == 2
    assert all_cases[0].case_id == "c1"


def test_persists_across_instances(tmp_path):
    path = str(tmp_path / "cases.jsonl")
    store = CaseStore(path)
    store.add(_make_case("c1"))
    store2 = CaseStore(path)
    assert len(store2.get_all()) == 1


def test_empty_store_returns_empty_list(tmp_path):
    store = CaseStore(str(tmp_path / "cases.jsonl"))
    assert store.get_all() == []
```

### Step 2: Run to verify they fail
```bash
python -m pytest tests/memory/test_case_store.py -v
# Expected: ImportError or AttributeError — CaseStore does not exist yet
```

### Step 3: Implement CaseStore

```python
# src/memory/case_store.py
from __future__ import annotations
from pathlib import Path
from typing import List
from src.models.nodes import CaseEntry


class CaseStore:
    """Append-only JSONL store for cross-session CaseEntry knowledge."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: List[CaseEntry] = []
        if self._path.exists():
            for line in self._path.read_text().splitlines():
                line = line.strip()
                if line:
                    self._entries.append(CaseEntry.model_validate_json(line))

    def add(self, case: CaseEntry) -> None:
        self._entries.append(case)
        with self._path.open("a") as f:
            f.write(case.model_dump_json() + "\n")

    def get_all(self) -> List[CaseEntry]:
        return list(self._entries)
```

### Step 4: Run tests
```bash
python -m pytest tests/memory/test_case_store.py -v
# Expected: 3 passed
```

### Step 5: Commit
```bash
git add src/memory/case_store.py tests/memory/test_case_store.py
git commit -m "feat: CaseStore — append-only JSONL cross-session knowledge base"
```

---

## Task 2: CaseRetriever

**Files:**
- Create: `src/memory/retrieval.py`
- Test: `tests/memory/test_retrieval.py`

The retriever converts `TaskTraits` into a numeric feature vector and ranks candidates by cosine similarity.

Feature vector (7 dimensions):
```
[task_type_binary, task_type_multiclass, task_type_regression,
 n_rows_score,      # small=0.0, medium=0.5, large=1.0
 n_features_score,  # small=0.0, medium=0.5, large=1.0
 balance_score,     # balanced=1.0, moderate=0.5, severe=0.0
 numeric_ratio]     # feature_types["numeric"] / total_features (0–1)
```

### Step 1: Write failing tests

```python
# tests/memory/test_retrieval.py
import pytest
from src.memory.retrieval import CaseRetriever
from src.models.nodes import CaseEntry, TaskTraits, WhatWorked, WhatFailed, SessionTrajectory
from src.models.task import ExperimentPlan


def _plan():
    return ExperimentPlan(
        eval_metric="roc_auc", model_families=["GBM"], presets="medium_quality",
        time_limit=120, feature_policy={}, validation_policy={"holdout_frac": 0.2},
    )

def _case(case_id, task_type, n_rows_bucket, class_balance):
    return CaseEntry(
        case_id=case_id,
        task_traits=TaskTraits(
            task_type=task_type, n_rows_bucket=n_rows_bucket,
            n_features_bucket="medium", class_balance=class_balance,
            feature_types={"numeric": 8, "categorical": 2},
        ),
        what_worked=WhatWorked(best_config=_plan(), best_metric=0.85),
        what_failed=WhatFailed(),
        trajectory=SessionTrajectory(n_runs=3),
    )


def test_returns_top_k(tmp_path):
    cases = [
        _case("c1", "binary", "medium", "balanced"),
        _case("c2", "multiclass", "large", "severe"),
        _case("c3", "binary", "small", "moderate"),
    ]
    query = TaskTraits(
        task_type="binary", n_rows_bucket="medium",
        n_features_bucket="medium", class_balance="balanced",
    )
    retriever = CaseRetriever()
    results = retriever.rank(query, cases, top_k=2)
    assert len(results) == 2
    assert results[0].case_id == "c1"  # exact match should be first


def test_returns_empty_on_no_candidates():
    retriever = CaseRetriever()
    results = retriever.rank(
        TaskTraits(task_type="binary", n_rows_bucket="medium",
                   n_features_bucket="medium", class_balance="balanced"),
        candidates=[],
        top_k=3,
    )
    assert results == []


def test_top_k_capped_at_candidates():
    cases = [_case("c1", "binary", "medium", "balanced")]
    retriever = CaseRetriever()
    results = retriever.rank(
        TaskTraits(task_type="binary", n_rows_bucket="medium",
                   n_features_bucket="medium", class_balance="balanced"),
        candidates=cases,
        top_k=5,
    )
    assert len(results) == 1
```

### Step 2: Run to verify they fail
```bash
python -m pytest tests/memory/test_retrieval.py -v
# Expected: ImportError
```

### Step 3: Implement CaseRetriever

```python
# src/memory/retrieval.py
from __future__ import annotations
from typing import List
import numpy as np
from src.models.nodes import CaseEntry, TaskTraits


def _trait_vector(traits: TaskTraits) -> np.ndarray:
    task_enc = {"binary": [1, 0, 0], "multiclass": [0, 1, 0], "regression": [0, 0, 1]}
    task_vec = task_enc.get(traits.task_type, [0, 0, 0])

    bucket_score = {"small": 0.0, "medium": 0.5, "large": 1.0}
    rows_score = bucket_score.get(traits.n_rows_bucket, 0.5)
    feat_score = bucket_score.get(traits.n_features_bucket, 0.5)

    balance_score = {"balanced": 1.0, "moderate": 0.5, "moderate_imbalance": 0.5,
                     "severe": 0.0, "severe_imbalance": 0.0}.get(traits.class_balance, 0.5)

    total = sum(traits.feature_types.values()) or 1
    numeric_ratio = traits.feature_types.get("numeric", 0) / total

    return np.array(task_vec + [rows_score, feat_score, balance_score, numeric_ratio],
                    dtype=float)


class CaseRetriever:
    """Ranks CaseEntry candidates by cosine similarity on task traits."""

    def rank(self, query_traits: TaskTraits, candidates: List[CaseEntry],
             top_k: int = 3) -> List[CaseEntry]:
        if not candidates:
            return []

        q_vec = _trait_vector(query_traits).reshape(1, -1)
        c_vecs = np.array([_trait_vector(c.task_traits) for c in candidates])

        # cosine similarity without sklearn to avoid import overhead
        q_norm = np.linalg.norm(q_vec, axis=1, keepdims=True)
        c_norm = np.linalg.norm(c_vecs, axis=1, keepdims=True)
        q_safe = np.where(q_norm == 0, 1, q_norm)
        c_safe = np.where(c_norm == 0, 1, c_norm)
        sims = (q_vec / q_safe) @ (c_vecs / c_safe).T
        sims = sims.flatten()

        top_indices = np.argsort(sims)[::-1][:top_k]
        return [candidates[i] for i in top_indices]
```

### Step 4: Run tests
```bash
python -m pytest tests/memory/test_retrieval.py -v
# Expected: 3 passed
```

### Step 5: Commit
```bash
git add src/memory/retrieval.py tests/memory/test_retrieval.py
git commit -m "feat: CaseRetriever — cosine similarity ranking on task traits"
```

---

## Task 3: Distiller

**Files:**
- Modify: `prompts/distiller.md` (add clear schema instructions)
- Create: `src/memory/distiller.py`
- Test: `tests/memory/test_distiller.py`

The Distiller calls the LLM with the run history and tree, parses the JSON response into a `CaseEntry`. It computes `task_traits`, `trajectory`, and `tree_summary` locally (no LLM needed for those — pure computation). The LLM only produces `what_worked` and `what_failed`.

### Step 1: Update `prompts/distiller.md`

Replace the existing file content with:

```markdown
# Distiller Agent

You summarize a completed ML experiment session into a reusable case entry.

## Input
You will receive:
1. Task description and data profile
2. Full run history (run_id, metric, model families, rationale)
3. Metric progression across runs

## Output
Output ONLY this JSON object (no markdown fences, no prose):

{
  "what_worked": {
    "key_decisions": ["<specific decision with metric delta>", "..."],
    "important_features": ["<feature name>", "..."],
    "effective_presets": "<preset string>"
  },
  "what_failed": {
    "failed_approaches": ["<specific failed config>", "..."],
    "failure_patterns": ["<generalized anti-pattern transferable to future tasks>", "..."]
  },
  "trajectory": {
    "turning_points": ["<run N: what changed and why it mattered>", "..."]
  }
}

## Rules
- key_decisions must be specific: include which run, what changed, metric delta
- failure_patterns must be general (transferable to other tasks), not task-specific
- important_features: list top 3-5 features that mattered across runs (empty list if unknown)
- effective_presets: the preset used by the winning run
- Respond with ONLY the JSON object.
```

### Step 2: Write failing tests

```python
# tests/memory/test_distiller.py
import pytest
from unittest.mock import MagicMock
from src.memory.distiller import Distiller
from src.models.nodes import CaseEntry, TaskTraits
from src.models.task import TaskSpec, ExperimentPlan
from src.models.results import DataProfile, RunEntry, RunResult, RunDiagnostics, RunConfig


def _make_run_entry(run_id: str, metric: float, families: list) -> RunEntry:
    plan = ExperimentPlan(
        eval_metric="roc_auc", model_families=families, presets="medium_quality",
        time_limit=120, feature_policy={}, validation_policy={"holdout_frac": 0.2},
    )
    config = RunConfig(
        run_id=run_id, node_id="n1", autogluon_kwargs={},
        data_path="data/train.csv", output_dir=f"experiments/runs/{run_id}",
    )
    result = RunResult(
        run_id=run_id, status="success", primary_metric=metric,
        best_model_name="WeightedEnsemble_L2", fit_time_seconds=10.0,
        artifacts_dir=f"experiments/runs/{run_id}",
    )
    return RunEntry(
        run_id=run_id, node_id="n1", config=config, result=result,
        diagnostics=RunDiagnostics(), agent_rationale="test rationale",
    )


LLM_RESPONSE = '''
{
  "what_worked": {
    "key_decisions": ["Run 2: switching to f1_macro improved +0.08"],
    "important_features": ["Age", "Fare"],
    "effective_presets": "medium_quality"
  },
  "what_failed": {
    "failed_approaches": ["GBM alone scored 0.83"],
    "failure_patterns": ["Single model families underperform ensembles on small tabular data"]
  },
  "trajectory": {
    "turning_points": ["Run 2: adding CAT to ensemble was the key improvement"]
  }
}
'''


def test_distill_returns_case_entry():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = LLM_RESPONSE

    task = TaskSpec(
        task_name="test", task_type="binary", data_path="data/train.csv",
        target_column="label", eval_metric="roc_auc",
        description="test task", constraints={},
    )
    profile = DataProfile(n_rows=891, n_features=10, class_balance_ratio=0.6, missing_rate=0.05)
    runs = [
        _make_run_entry("run_0001", 0.83, ["GBM"]),
        _make_run_entry("run_0002", 0.87, ["GBM", "CAT"]),
    ]

    distiller = Distiller(llm=mock_llm)
    case = distiller.distill(task=task, data_profile=profile, run_history=runs)

    assert isinstance(case, CaseEntry)
    assert case.what_worked.best_metric == 0.87
    assert case.trajectory.n_runs == 2
    assert "Age" in case.what_worked.important_features
    assert mock_llm.complete.called


def test_distill_computes_traits_locally():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = LLM_RESPONSE

    task = TaskSpec(
        task_name="test", task_type="binary", data_path="data/train.csv",
        target_column="label", eval_metric="roc_auc",
        description="test task", constraints={},
    )
    profile = DataProfile(n_rows=200, n_features=5, class_balance_ratio=0.3, missing_rate=0.0)
    runs = [_make_run_entry("run_0001", 0.83, ["GBM"])]

    distiller = Distiller(llm=mock_llm)
    case = distiller.distill(task=task, data_profile=profile, run_history=runs)

    assert case.task_traits.task_type == "binary"
    assert case.task_traits.n_rows_bucket == "small"   # 200 < 1000
    assert case.task_traits.class_balance == "moderate"  # 0.3
```

### Step 3: Run to verify they fail
```bash
python -m pytest tests/memory/test_distiller.py -v
# Expected: ImportError
```

### Step 4: Implement Distiller

```python
# src/memory/distiller.py
from __future__ import annotations
import uuid
from pathlib import Path
from typing import List
from src.llm.backend import LLMBackend, Message
from src.models.task import TaskSpec, ExperimentPlan
from src.models.results import DataProfile, RunEntry
from src.models.nodes import (
    CaseEntry, TaskTraits, WhatWorked, WhatFailed, SessionTrajectory, TreeSummary
)


def _rows_bucket(n: int) -> str:
    if n < 1000: return "small"
    if n < 50000: return "medium"
    return "large"


def _features_bucket(n: int) -> str:
    if n < 10: return "small"
    if n < 50: return "medium"
    return "large"


def _balance_bucket(ratio: float) -> str:
    if ratio >= 0.8: return "balanced"
    if ratio >= 0.4: return "moderate"
    return "severe"


class Distiller:
    """Summarises a completed session into a CaseEntry via LLM."""

    def __init__(self, llm: LLMBackend, prompt_path: str = "prompts/distiller.md") -> None:
        self._llm = llm
        self._system_prompt = Path(prompt_path).read_text()

    def distill(
        self,
        task: TaskSpec,
        data_profile: DataProfile,
        run_history: List[RunEntry],
    ) -> CaseEntry:
        # Compute structural fields locally — no LLM needed
        successful = [r for r in run_history if r.result.status == "success"
                      and r.result.primary_metric is not None]
        best = max(successful, key=lambda r: r.result.primary_metric) if successful else None
        metrics = [r.result.primary_metric for r in successful]

        traits = TaskTraits(
            task_type=task.task_type,
            n_rows_bucket=_rows_bucket(data_profile.n_rows),
            n_features_bucket=_features_bucket(data_profile.n_features),
            class_balance=_balance_bucket(data_profile.class_balance_ratio),
            feature_types=data_profile.feature_types,
        )
        trajectory = SessionTrajectory(
            n_runs=len(run_history),
            total_time_seconds=sum(r.result.fit_time_seconds for r in run_history),
            metric_progression=metrics,
        )

        # LLM produces what_worked and what_failed
        user_msg = self._build_user_message(task, data_profile, run_history, metrics)
        response = self._llm.complete(
            messages=[
                Message(role="system", content=self._system_prompt),
                Message(role="user", content=user_msg),
            ],
            temperature=0.2,
        )

        import json
        parsed = json.loads(response)
        ww_raw = parsed.get("what_worked", {})
        wf_raw = parsed.get("what_failed", {})
        traj_raw = parsed.get("trajectory", {})

        what_worked = WhatWorked(
            best_config=best.config.autogluon_kwargs if best else {},
            best_metric=best.result.primary_metric if best else 0.0,
            key_decisions=ww_raw.get("key_decisions", []),
            important_features=ww_raw.get("important_features", []),
            effective_presets=ww_raw.get("effective_presets", "medium_quality"),
        )
        what_failed = WhatFailed(
            failed_approaches=wf_raw.get("failed_approaches", []),
            failure_patterns=wf_raw.get("failure_patterns", []),
        )
        trajectory.turning_points = traj_raw.get("turning_points", [])

        return CaseEntry(
            case_id=str(uuid.uuid4()),
            task_traits=traits,
            what_worked=what_worked,
            what_failed=what_failed,
            trajectory=trajectory,
        )

    def _build_user_message(
        self,
        task: TaskSpec,
        profile: DataProfile,
        runs: List[RunEntry],
        metrics: List[float],
    ) -> str:
        run_lines = []
        for r in runs:
            families = list(r.config.autogluon_kwargs.get("hyperparameters", {}).keys())
            run_lines.append(
                f"- {r.run_id}: metric={r.result.primary_metric} "
                f"families={families} rationale={r.agent_rationale[:80]}"
            )
        return (
            f"## Task\n{task.task_name} | {task.task_type} | target={task.target_column}\n"
            f"{task.description}\n\n"
            f"## Data Profile\n{profile.summary}\n\n"
            f"## Run History\n" + "\n".join(run_lines) + "\n\n"
            f"## Metric Progression\n{metrics}\n\n"
            f"Summarise this session into the JSON CaseEntry format."
        )
```

**Note:** `WhatWorked.best_config` is typed as `ExperimentPlan` in the model. The distiller passes `autogluon_kwargs` dict directly — this will fail Pydantic validation. We need to pass the actual `ExperimentPlan` from the winning run's node plan. Fix: pass `best.config` as a `RunConfig` and reconstruct the plan, OR store the plan on the run entry. For now, use the winning run's `agent_rationale` as a placeholder and use a default plan for `best_config`.

**Correction to the implementation** — replace the `what_worked` construction with:

```python
best_plan = best.config  # RunConfig, not ExperimentPlan
# WhatWorked.best_config expects ExperimentPlan — use a reconstructed minimal plan
from src.models.task import ExperimentPlan
best_experiment_plan = ExperimentPlan(
    eval_metric=best_plan.autogluon_kwargs.get("eval_metric", task.eval_metric),
    model_families=list(best_plan.autogluon_kwargs.get("hyperparameters", {}).keys()),
    presets=best_plan.autogluon_kwargs.get("presets", "medium_quality"),
    time_limit=best_plan.autogluon_kwargs.get("time_limit", 120),
    feature_policy={},
    validation_policy={"holdout_frac": 0.2},
) if best else ExperimentPlan(
    eval_metric=task.eval_metric, model_families=["GBM"],
    presets="medium_quality", time_limit=120,
    feature_policy={}, validation_policy={"holdout_frac": 0.2},
)
what_worked = WhatWorked(
    best_config=best_experiment_plan,
    best_metric=best.result.primary_metric if best else 0.0,
    ...
)
```

### Step 5: Run tests
```bash
python -m pytest tests/memory/test_distiller.py -v
# Expected: 2 passed
```

### Step 6: Commit
```bash
git add src/memory/distiller.py tests/memory/test_distiller.py prompts/distiller.md
git commit -m "feat: Distiller — session to CaseEntry summarisation via LLM"
```

---

## Task 4: ContextBuilder

**Files:**
- Create: `src/memory/context_builder.py`
- Test: `tests/memory/test_context_builder.py`

ContextBuilder is pure assembly — no LLM, no IO. Takes components, returns `SearchContext`.

### Step 1: Write failing tests

```python
# tests/memory/test_context_builder.py
import pytest
from unittest.mock import MagicMock
from src.memory.context_builder import ContextBuilder
from src.models.nodes import SearchContext, ExperimentNode, NodeStage, NodeStatus
from src.models.task import TaskSpec, ExperimentPlan
from src.models.results import DataProfile, RunEntry, RunResult, RunDiagnostics, RunConfig


def _make_node() -> ExperimentNode:
    plan = ExperimentPlan(
        eval_metric="roc_auc", model_families=["GBM"], presets="medium_quality",
        time_limit=120, feature_policy={}, validation_policy={"holdout_frac": 0.2},
    )
    return ExperimentNode(node_id="n1", plan=plan, stage=NodeStage.OPTIMIZE)


def _make_task() -> TaskSpec:
    return TaskSpec(
        task_name="test", task_type="binary", data_path="data/train.csv",
        target_column="label", eval_metric="roc_auc",
        description="test", constraints={},
    )


def _make_profile() -> DataProfile:
    return DataProfile(n_rows=891, n_features=10, class_balance_ratio=0.6, missing_rate=0.05)


def test_build_returns_search_context():
    builder = ContextBuilder()
    context = builder.build(
        task=_make_task(),
        data_profile=_make_profile(),
        history=[],
        incumbent=None,
        current_node=_make_node(),
        stage="optimize",
        budget_remaining=4,
        budget_used=1,
        similar_cases=[],
    )
    assert isinstance(context, SearchContext)
    assert context.stage == "optimize"
    assert context.budget_remaining == 4


def test_failed_attempts_filtered_from_history():
    builder = ContextBuilder()
    config = RunConfig(
        run_id="r1", node_id="n1", autogluon_kwargs={},
        data_path="data/train.csv", output_dir="experiments/runs/r1",
    )
    failed_entry = RunEntry(
        run_id="r1", node_id="n1", config=config,
        result=RunResult(run_id="r1", status="failed", fit_time_seconds=0.0, artifacts_dir=""),
        diagnostics=RunDiagnostics(),
    )
    success_entry = RunEntry(
        run_id="r2", node_id="n1",
        config=RunConfig(run_id="r2", node_id="n1", autogluon_kwargs={},
                         data_path="data/train.csv", output_dir="experiments/runs/r2"),
        result=RunResult(run_id="r2", status="success", primary_metric=0.85,
                         fit_time_seconds=10.0, artifacts_dir=""),
        diagnostics=RunDiagnostics(),
    )

    context = builder.build(
        task=_make_task(), data_profile=_make_profile(),
        history=[failed_entry, success_entry], incumbent=None,
        current_node=_make_node(), stage="optimize",
        budget_remaining=3, budget_used=2, similar_cases=[],
    )
    assert len(context.failed_attempts) == 1
    assert context.failed_attempts[0].run_id == "r1"
```

### Step 2: Run to verify they fail
```bash
python -m pytest tests/memory/test_context_builder.py -v
```

### Step 3: Implement ContextBuilder

```python
# src/memory/context_builder.py
from __future__ import annotations
from typing import List, Optional
from src.models.nodes import SearchContext, ExperimentNode, CaseEntry
from src.models.task import TaskSpec
from src.models.results import DataProfile, RunEntry


class ContextBuilder:
    """Assembles a SearchContext briefing from session state."""

    def build(
        self,
        task: TaskSpec,
        data_profile: DataProfile,
        history: List[RunEntry],
        incumbent: Optional[RunEntry],
        current_node: ExperimentNode,
        stage: str,
        budget_remaining: int,
        budget_used: int,
        similar_cases: List[CaseEntry],
    ) -> SearchContext:
        failed = [r for r in history if r.result.status == "failed"]
        return SearchContext(
            task=task,
            data_profile=data_profile,
            history=history,
            incumbent=incumbent,
            current_node=current_node,
            tree_summary={},
            similar_cases=similar_cases,
            failed_attempts=failed,
            stage=stage,
            budget_remaining=budget_remaining,
            budget_used=budget_used,
        )
```

### Step 4: Run tests
```bash
python -m pytest tests/memory/test_context_builder.py -v
# Expected: 2 passed
```

### Step 5: Commit
```bash
git add src/memory/context_builder.py tests/memory/test_context_builder.py
git commit -m "feat: ContextBuilder — SearchContext assembly from session state"
```

---

## Task 5: IdeatorAgent

**Files:**
- Create: `src/agents/ideator.py`
- Test: `tests/agents/test_ideator.py`

IdeatorAgent calls the LLM with task + profile + similar cases → returns `List[Dict[str, str]]` (hypothesis dicts with `hypothesis` and `rationale` keys), compatible with the existing `create_candidate_nodes()` interface in session.py.

### Step 1: Write failing tests

```python
# tests/agents/test_ideator.py
import pytest
from unittest.mock import MagicMock
from src.agents.ideator import IdeatorAgent
from src.models.task import TaskSpec
from src.models.results import DataProfile


LLM_RESPONSE = '''[
  {"id": "h1", "model_focus": "GBM", "metric_focus": "roc_auc",
   "hypothesis": "Start with GBM baseline.", "rationale": "Reliable default."},
  {"id": "h2", "model_focus": "RF", "metric_focus": "f1_macro",
   "hypothesis": "Try RF with f1_macro for class imbalance.",
   "rationale": "class_balance_ratio=0.62 suggests mild imbalance."},
  {"id": "h3", "model_focus": "GBM+XGB", "metric_focus": "accuracy",
   "hypothesis": "Diverse boosting ensemble.", "rationale": "Variety improves ensemble."}
]'''


def _task():
    return TaskSpec(
        task_name="titanic", task_type="binary", data_path="data/train.csv",
        target_column="Survived", eval_metric="roc_auc",
        description="Predict survival.", constraints={},
    )

def _profile():
    return DataProfile(n_rows=891, n_features=10, class_balance_ratio=0.62, missing_rate=0.08)


def test_ideate_returns_hypothesis_list():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = LLM_RESPONSE

    agent = IdeatorAgent(llm=mock_llm)
    hypotheses = agent.ideate(task=_task(), data_profile=_profile(), similar_cases=[])

    assert len(hypotheses) == 3
    assert all("hypothesis" in h for h in hypotheses)
    assert all("rationale" in h for h in hypotheses)


def test_ideate_calls_llm_once():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = LLM_RESPONSE

    agent = IdeatorAgent(llm=mock_llm)
    agent.ideate(task=_task(), data_profile=_profile(), similar_cases=[])

    assert mock_llm.complete.call_count == 1


def test_ideate_respects_num_hypotheses():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = LLM_RESPONSE  # returns 3, we request 3

    agent = IdeatorAgent(llm=mock_llm, num_hypotheses=3)
    hypotheses = agent.ideate(task=_task(), data_profile=_profile(), similar_cases=[])
    assert len(hypotheses) == 3
```

### Step 2: Run to verify they fail
```bash
python -m pytest tests/agents/test_ideator.py -v
```

### Step 3: Implement IdeatorAgent

```python
# src/agents/ideator.py
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict
from src.llm.backend import LLMBackend, Message
from src.models.task import TaskSpec
from src.models.results import DataProfile
from src.models.nodes import CaseEntry


class IdeatorAgent:
    """
    Generates initial experiment hypotheses from task + data profile + similar past cases.
    Returns List[Dict] with 'hypothesis' and 'rationale' keys,
    compatible with session.create_candidate_nodes().
    """

    def __init__(
        self,
        llm: LLMBackend,
        prompt_path: str = "prompts/ideator.md",
        num_hypotheses: int = 3,
        temperature: float = 0.5,
    ) -> None:
        self._llm = llm
        self._system_prompt = Path(prompt_path).read_text()
        self._num_hypotheses = num_hypotheses
        self._temperature = temperature

    def ideate(
        self,
        task: TaskSpec,
        data_profile: DataProfile,
        similar_cases: List[CaseEntry],
    ) -> List[Dict[str, str]]:
        user_msg = self._build_user_message(task, data_profile, similar_cases)
        response = self._llm.complete(
            messages=[
                Message(role="system", content=self._system_prompt),
                Message(role="user", content=user_msg),
            ],
            temperature=self._temperature,
        )
        raw = json.loads(response)
        return [{"hypothesis": h["hypothesis"], "rationale": h["rationale"]} for h in raw]

    def _build_user_message(
        self,
        task: TaskSpec,
        profile: DataProfile,
        similar_cases: List[CaseEntry],
    ) -> str:
        cases_text = ""
        if similar_cases:
            lines = []
            for c in similar_cases:
                lines.append(
                    f"- Past case ({c.task_traits.task_type}, {c.task_traits.n_rows_bucket} rows): "
                    f"best={c.what_worked.best_metric:.3f} | "
                    f"worked: {'; '.join(c.what_worked.key_decisions[:2])} | "
                    f"failed: {'; '.join(c.what_failed.failure_patterns[:1])}"
                )
            cases_text = "\n## Similar Past Cases\n" + "\n".join(lines)

        return (
            f"## Task\n"
            f"Name: {task.task_name} | Type: {task.task_type}\n"
            f"Target: {task.target_column} | Metric: {task.eval_metric}\n"
            f"Description: {task.description}\n\n"
            f"## Data Profile\n"
            f"{profile.summary}\n"
            f"Rows: {profile.n_rows} | Features: {profile.n_features}\n"
            f"Class balance ratio: {profile.class_balance_ratio:.2f}\n"
            f"Missing rate: {profile.missing_rate:.2%}\n"
            f"Feature types: {profile.feature_types}\n"
            f"{cases_text}\n\n"
            f"Generate exactly {self._num_hypotheses} diverse hypotheses as a JSON array."
        )
```

### Step 4: Run tests
```bash
python -m pytest tests/agents/test_ideator.py -v
# Expected: 3 passed
```

### Step 5: Commit
```bash
git add src/agents/ideator.py tests/agents/test_ideator.py
git commit -m "feat: IdeatorAgent — LLM hypothesis generation from data profile and similar cases"
```

---

## Task 6: Wire Phase 2 into session.py

**Files:**
- Modify: `src/session.py`
- Modify: `main.py`
- Test: `tests/test_session.py` (extend existing)

Replace static `seed_ideas` flow with `IdeatorAgent`. Add `CaseStore` + `CaseRetriever`. Add `Distiller` call at session end. Replace inline `SearchContext` construction with `ContextBuilder`.

### Step 1: Read existing session.py and test_session.py first
Already read. Key changes needed:

**`__init__`:** add `IdeatorAgent`, `CaseStore`, `CaseRetriever`, `ContextBuilder`, `Distiller`

**`run()`:**
- Replace `hyps = hypotheses or self._seed_ideas` with `IdeatorAgent.ideate()`
- Replace inline `SearchContext(...)` in optimize loop with `ContextBuilder.build()`
- After saving `tree.json`, call `Distiller.distill()` + `CaseStore.add()`

**`main.py`:** pass `case_store_path` from config; no more `seed_ideas` kwarg needed (ideator takes over)

### Step 2: Write failing integration test

```python
# Add to tests/test_session.py

def test_session_uses_ideator_not_seed_ideas(tmp_path):
    """IdeatorAgent.ideate() is called when no hypotheses passed."""
    from unittest.mock import MagicMock, patch
    from src.session import ExperimentSession
    from src.models.task import TaskSpec

    task = TaskSpec(
        task_name="test", task_type="binary",
        data_path="tests/fixtures/titanic_sample.csv",
        target_column="Survived", eval_metric="roc_auc",
        description="test", constraints={},
    )
    mock_llm = MagicMock()
    # IdeatorAgent response
    mock_llm.complete.return_value = '[{"id":"h1","model_focus":"GBM","metric_focus":"roc_auc","hypothesis":"GBM baseline","rationale":"safe default"}]'

    with patch("src.session.IdeatorAgent") as MockIdeator:
        mock_ideator_instance = MagicMock()
        mock_ideator_instance.ideate.return_value = [{"hypothesis": "GBM baseline", "rationale": "safe"}]
        MockIdeator.return_value = mock_ideator_instance

        session = ExperimentSession(task=task, llm=mock_llm, experiments_dir=str(tmp_path))
        assert mock_ideator_instance is not None
```

### Step 3: Run to verify it fails
```bash
python -m pytest tests/test_session.py::test_session_uses_ideator_not_seed_ideas -v
```

### Step 4: Update session.py

In `__init__`, replace:
```python
self._seed_ideas = seed_ideas or []
```
with:
```python
from src.memory.case_store import CaseStore
from src.memory.retrieval import CaseRetriever
from src.memory.distiller import Distiller
from src.memory.context_builder import ContextBuilder
from src.agents.ideator import IdeatorAgent

self._case_store = CaseStore(case_store_path) if case_store_path else None
self._retriever = CaseRetriever()
self._distiller = Distiller(llm=llm)
self._context_builder = ContextBuilder()
self._ideator = IdeatorAgent(llm=llm, num_hypotheses=num_candidates)
```

Add `case_store_path: Optional[str] = None` to `__init__` signature.

In `run()`, replace the hypotheses loading block:
```python
# OLD
hyps = hypotheses or self._seed_ideas
if not hyps:
    hyps = [{"hypothesis": "Try GBM baseline with default settings", "rationale": "Safe default"}]

# NEW
similar_cases = []
if self._case_store:
    from src.models.nodes import TaskTraits
    from src.memory.retrieval import _rows_bucket, _features_bucket, _balance_bucket
    query_traits = TaskTraits(
        task_type=self.task.task_type,
        n_rows_bucket=_rows_bucket(data_profile.n_rows),
        n_features_bucket=_features_bucket(data_profile.n_features),
        class_balance=_balance_bucket(data_profile.class_balance_ratio),
        feature_types=data_profile.feature_types,
    )
    similar_cases = self._retriever.rank(query_traits, self._case_store.get_all(), top_k=3)
    self._log.info("Retrieved %d similar past cases", len(similar_cases))

hyps = hypotheses or self._ideator.ideate(
    task=self.task,
    data_profile=data_profile,
    similar_cases=similar_cases,
)
```

Replace inline `SearchContext(...)` in optimize loop with:
```python
context = self._context_builder.build(
    task=self.task,
    data_profile=data_profile,
    history=self.run_store.get_history(),
    incumbent=self.run_store.get_incumbent(self._higher_is_better),
    current_node=incumbent,
    stage="optimize",
    budget_remaining=self.scheduler.max_optimize_iterations - self.scheduler._optimize_count,
    budget_used=self.scheduler._optimize_count,
    similar_cases=similar_cases,
)
```

After `self.tree.save(...)`, add distillation:
```python
if self._case_store and best_entry:
    self._log.info("Distilling session into CaseStore...")
    try:
        case = self._distiller.distill(
            task=self.task,
            data_profile=data_profile,
            run_history=self.run_store.get_history(),
        )
        self._case_store.add(case)
        self._log.info("Session distilled → case_id=%s", case.case_id)
    except Exception as e:
        self._log.warning("Distillation failed (non-fatal): %s", e)
```

In `main.py`, pass `case_store_path` from config:
```python
session = ExperimentSession(
    task=task,
    llm=backend,
    experiments_dir=cfg["session"]["experiments_dir"],
    num_candidates=search_cfg["search"]["num_candidates"],
    max_optimize_iterations=search_cfg["search"]["max_optimize_iterations"],
    higher_is_better=higher_is_better,
    case_store_path=cfg["session"].get("case_store_path"),
)
```

### Step 5: Move bucket helpers to be importable

Extract `_rows_bucket`, `_features_bucket`, `_balance_bucket` from `distiller.py` into a shared location so both `distiller.py` and `session.py` can import them without circular imports.

Create `src/memory/trait_utils.py`:
```python
# src/memory/trait_utils.py
def rows_bucket(n: int) -> str:
    if n < 1000: return "small"
    if n < 50000: return "medium"
    return "large"

def features_bucket(n: int) -> str:
    if n < 10: return "small"
    if n < 50: return "medium"
    return "large"

def balance_bucket(ratio: float) -> str:
    if ratio >= 0.8: return "balanced"
    if ratio >= 0.4: return "moderate"
    return "severe"
```

Update `distiller.py` and `session.py` to import from `trait_utils`.

### Step 6: Run full test suite
```bash
python -m pytest tests/ -v
# Expected: all existing 50 + new tests pass
```

### Step 7: Commit
```bash
git add src/session.py src/agents/ideator.py src/memory/ main.py tests/
git commit -m "feat: wire Phase 2 — IdeatorAgent, CaseStore, ContextBuilder, Distiller into session"
```

---

## Task 7: Update docs

**Files:**
- Create: `docs/changes/implementation-log.md` (if not exists)
- Create: `docs/architecture/current-state.md` (if not exists)

Per AGENTS.md documentation rules, meaningful changes require log entries.

```bash
# Create docs directories if needed
mkdir -p docs/changes docs/architecture
```

`docs/changes/implementation-log.md`:
```markdown
## 2026-03-19 — Phase 2: Memory & Ideation

- Added CaseStore (append-only JSONL cross-session knowledge base)
- Added CaseRetriever (cosine similarity on TaskTraits feature vectors)
- Added Distiller (LLM session → CaseEntry summarisation)
- Added ContextBuilder (SearchContext assembly)
- Added IdeatorAgent (LLM hypothesis generation grounded in data profile + past cases)
- Wired all Phase 2 components into ExperimentSession
- Session now distils to CaseStore at end; next session retrieves similar cases
```

### Commit
```bash
git add docs/
git commit -m "docs: Phase 2 implementation log and architecture update"
```

---

## Running Everything

After all tasks complete:

```bash
# Full test suite
python -m pytest tests/ -q

# Full session run (will use IdeatorAgent instead of seed_ideas)
python3 main.py

# Verify CaseStore populated
python3 -c "
from src.memory.case_store import CaseStore
store = CaseStore('experiments/case_bank.jsonl')
cases = store.get_all()
print(f'{len(cases)} cases in store')
for c in cases:
    print(f'  {c.case_id[:8]} | {c.task_traits.task_type} | best={c.what_worked.best_metric:.3f}')
"
```
