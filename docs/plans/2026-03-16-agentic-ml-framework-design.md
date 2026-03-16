# Hybrid Agentic ML Framework — Architecture Design

**Date:** 2026-03-16
**Status:** Draft
**Approach:** Layered from Day One (Approach B), breadth-first build order
**Goal:** Learning exercise — understand agentic ML experiment loop architecture
**LLM:** Multi-provider (Anthropic + OpenAI), thin abstraction, no framework dependency

---

## 1. Core Concept

A guarded, closed-loop ML operating system that iteratively improves model performance
by combining LLM-based reasoning (the agent) with AutoML execution (AutoGluon),
structured experiment memory (run store + case store), and a graph-structured experiment
tree that tracks lineage, rationale, and diagnostics for every decision.

The agent does NOT replace AutoGluon's internal optimization. AutoGluon handles model
selection, ensembling, bagging, and stacking. The agent operates ABOVE AutoGluon,
reasoning about problem framing — which metric, which features, which validation strategy,
which model families to allow, how much budget to allocate.

### What makes it "agentic"

- The agent proposes structured experiment configs, not code
- Each run is diagnosed and the agent reviews results before deciding next steps
- Accept/reject mechanism prevents the agent from making things worse
- Experiment history (RunStore) + cross-session knowledge (CaseStore) give the agent
  grounded context for decisions, not blind search
- The ExperimentNode tree preserves full lineage: what was tried, what changed, what
  improved, and why — designed for future Graph RAG indexing

---

## 2. Architecture Overview

```
Task + Data
  |
  v
IDEATION (agent + case retrieval + data profiling)
  |
  v
EXPERIMENT LOOP
  |-- Agent proposes ExperimentPlan (structured JSON)
  |-- Orchestration creates ExperimentNode, maps to RunConfig
  |-- Executor runs AutoGluon
  |-- Result parser produces RunResult + diagnostics
  |-- RunEntry (config + result + diagnostics + rationale) stored in RunStore
  |-- Agent reviews RunEntry, orchestration accepts/rejects vs incumbent
  |-- Context builder assembles SearchContext for next iteration
  |-- Repeat until budget exhausted or plateau detected
  |
  v
DISTILLATION (summarize session -> CaseEntry -> CaseStore)
```

### Four Layers

```
+-----------------------------------------------------------+
|  AGENT LAYER                                              |
|  Decides WHAT to try next                                 |
|  ideator, manager, selector, refiner, reviewer            |
|  Input: SearchContext    Output: ExperimentPlan / Action   |
+-----------------------------------------------------------+
|  ORCHESTRATION LAYER                                      |
|  Controls HOW experiments branch and budget                |
|  ExperimentNode tree, scheduler, accept/reject             |
|  Input: ExperimentPlan   Output: RunConfig + node mgmt    |
+-----------------------------------------------------------+
|  EXECUTION LAYER                                          |
|  Runs ML (boring and reliable)                            |
|  AutoGluon runner, config mapper, result parser            |
|  Input: RunConfig        Output: RunResult + diagnostics  |
+-----------------------------------------------------------+
|  MEMORY LAYER                                             |
|  Stores what happened, retrieves relevant history          |
|  RunStore, CaseStore, distiller, retrieval, context builder|
|  Input: RunEntry         Output: SearchContext             |
+-----------------------------------------------------------+
```

---

## 3. Data Objects

All defined as dataclasses or Pydantic models in `src/models/`.

### 3.1 TaskSpec (models/task.py)

The problem definition. Created from `configs/project.yaml` + data inspection.

```
TaskSpec:
  task_name: str                     # "customer-churn"
  task_type: str                     # "binary", "multiclass", "regression"
  data_path: str                     # path to training data
  target_column: str
  eval_metric: str                   # initial metric (agent may change this)
  constraints: Dict                  # max_time, forbidden_models, etc.
  description: str                   # natural language task description
```

### 3.2 DataProfile (models/results.py)

Structured dataset statistics produced by profiling. Stored once per session,
referenced by ideator and reviewer agents.

```
DataProfile:
  n_rows: int
  n_features: int
  feature_types: Dict[str, int]      # {numeric: 15, categorical: 8}
  target_distribution: Dict          # class counts or value stats
  class_balance_ratio: float         # min_class / max_class
  missing_rate: float                # fraction of missing values
  high_cardinality_cols: List[str]   # categoricals with many unique values
  suspected_leakage_cols: List[str]  # features too correlated with target
  summary: str                       # natural language summary
```

### 3.3 ExperimentPlan (models/task.py)

The agent's proposed approach. Structured JSON, never free-form code.

```
ExperimentPlan:
  eval_metric: str                   # "roc_auc", "f1_macro", "rmse"
  model_families: List[str]          # ["GBM", "XGB", "CAT", "NN_TORCH"]
  presets: str                       # "medium_quality", "best_quality"
  time_limit: int                    # seconds for AutoGluon
  feature_policy: Dict               # include/exclude columns
  validation_policy: Dict            # holdout_frac, num_bag_folds, group_col
  hyperparameters: Optional[Dict]    # override AutoGluon defaults
  use_fit_extra: bool                # extend existing predictor vs fresh fit
  rationale: str                     # why this plan, grounded in prior results
```

### 3.4 RunConfig (models/task.py)

Concrete execution config. Produced by ConfigMapper from ExperimentPlan.
Maps directly to `TabularPredictor.fit()` kwargs.

```
RunConfig:
  run_id: str
  node_id: str                       # which ExperimentNode this belongs to
  autogluon_kwargs: Dict             # exact kwargs for Predictor.fit()
  data_path: str
  output_dir: str                    # experiments/{session}/runs/run_XXXX/
```

### 3.5 RunResult (models/results.py)

What comes back from execution.

```
RunResult:
  run_id: str
  status: "success" | "failed"
  primary_metric: Optional[float]
  leaderboard: List[ModelEntry]      # top models with scores + fit times
  best_model_name: Optional[str]
  fit_time_seconds: float
  artifacts_dir: str                 # saved predictor, plots
  error: Optional[str]              # if failed, why
  raw_info: Dict                     # predictor.info() for deep inspection

ModelEntry:
  model_name: str
  score_val: float
  fit_time: float
  pred_time: float
  stack_level: int
```

### 3.6 RunEntry (models/results.py)

A RunResult enriched with diagnostics and agent reasoning. This is what gets
stored in RunStore — the complete record of one experiment.

```
RunEntry:
  run_id: str
  node_id: str
  timestamp: datetime
  config: RunConfig
  result: RunResult
  diagnostics:
    data_profile_ref: str            # reference to session's DataProfile
    overfitting_gap: Optional[float] # train vs val score difference
    feature_importances: Dict        # top features and their importance
    metric_vs_parent: Optional[float]# delta from parent node's metric
    change_description: str          # what changed from parent ("switched metric to f1")
    failure_mode: Optional[str]      # "class_imbalance", "leakage", "timeout", etc.
  agent_rationale: str               # why the agent proposed this
  agent_review: str                  # agent's assessment after seeing results
```

### 3.7 ExperimentNode (models/nodes.py)

A node in the experiment tree. Preserves full lineage for future Graph RAG.

```
ExperimentNode:
  node_id: str
  parent_id: Optional[str]          # None = root/draft node
  children: List[str]               # child node IDs
  edge_label: Optional[str]         # what changed from parent (graph edge metadata)
  stage: "ideation" | "warmup" | "optimize" | "debug"
  status: "pending" | "running" | "success" | "failed"
  plan: ExperimentPlan
  config: Optional[RunConfig]
  entry: Optional[RunEntry]         # full result + diagnostics
  depth: int                        # distance from root
  debug_depth: int                  # consecutive debug attempts
  created_at: datetime
```

Example tree:

```
Node A: "GBM + accuracy + all features + holdout"    [root, warmup]
|-- edge: "changed metric due to class imbalance"
+-- Node B: "GBM + f1_macro + all features + holdout"    [warmup]
    |-- edge: "dropped leaky columns, added k-fold"
    +-- Node C: "GBM + f1_macro + clean features + 5-fold"    [optimize]
        |-- edge: "increased budget"
        +-- Node D: "GBM + f1_macro + clean features + 5-fold + best_quality"  [optimize, INCUMBENT]

Node E: "XGB + accuracy + all features + holdout"    [root, warmup]
+-- (abandoned — worse than B's branch)

Node F: "NN_TORCH + accuracy + all features + holdout"    [root, warmup]
+-- Node G: "debug: reduce batch size for small data"    [debug, failed]
```

### 3.8 CaseEntry (models/nodes.py)

Distilled knowledge from a completed session. Stored in CaseStore for
cross-session retrieval. Designed for future Graph RAG indexing.

```
CaseEntry:
  case_id: str
  timestamp: datetime
  task_traits:
    task_type: str                   # "binary", "multiclass", "regression"
    n_rows_bucket: str               # "small", "medium", "large"
    n_features_bucket: str
    class_balance: str               # "balanced", "moderate_imbalance", "severe_imbalance"
    feature_types: Dict[str, int]
    domain_tags: List[str]           # user-provided or inferred
  what_worked:
    best_config: ExperimentPlan
    best_metric: float
    key_decisions: List[str]         # "switching to F1 improved +0.08"
    important_features: List[str]
    effective_presets: str
  what_failed:
    failed_approaches: List[str]     # "NN_TORCH overfitted with < 1k rows"
    failure_patterns: List[str]      # reusable anti-patterns
  trajectory:
    n_runs: int
    total_time_seconds: float
    metric_progression: List[float]  # metric at each iteration
    turning_points: List[str]        # key moments that changed direction
  tree_summary:                      # graph structure metadata for future Graph RAG
    n_nodes: int
    n_branches: int
    max_depth: int
    winning_path: List[str]          # node IDs from root to incumbent
    edge_labels_on_winning_path: List[str]
  embedding: Optional[List[float]]   # for similarity retrieval
```

### 3.9 SearchContext (models/nodes.py)

The "briefing document" assembled for the agent before each decision.

```
SearchContext:
  task: TaskSpec
  data_profile: DataProfile
  history: List[RunEntry]            # all prior runs this session
  incumbent: Optional[RunEntry]      # current best
  current_node: ExperimentNode       # where we are in the tree
  tree_summary: Dict                 # n_nodes, branches, depths
  similar_cases: List[CaseEntry]     # from CaseStore retrieval
  failed_attempts: List[RunEntry]    # what didn't work and why
  stage: str                         # "ideation", "warmup", "optimize"
  budget_remaining: int              # iterations left
  budget_used: int
```

---

## 4. Layer Details

### 4.1 Agent Layer

```
src/agents/
  manager.py       — ExperimentManager: top-level decision routing
  ideator.py       — data profiling + case retrieval -> initial hypotheses
  selector.py      — refines a hypothesis into a concrete ExperimentPlan
  refiner.py       — proposes config changes given RunEntry diagnostics
  reviewer.py      — assesses run quality, flags issues
```

**ExperimentManager** decides which sub-agent to invoke based on stage:

```
next_action(context: SearchContext) -> Action

Actions:
  Ideate(hypotheses)          # generate initial candidates
  Select(plan)                # turn hypothesis into ExperimentPlan
  Refine(changes)             # modify config based on results
  Debug(node, error_info)     # fix a failed run
  Accept(node)                # promote to incumbent
  Reject(node)                # revert to parent
  Stop(reason)                # session complete
```

**Ideator** runs once at session start:
1. Profiles the dataset -> DataProfile
2. Queries CaseStore for similar past tasks
3. Reads seed_ideas.json if available
4. Proposes K candidate hypotheses (default 3)

Each hypothesis becomes a root ExperimentNode.

**Key constraint:** agents output structured ExperimentPlan JSON, never Python code.
Prompts live in `prompts/` as markdown, loaded at runtime.

### 4.2 Orchestration Layer

```
src/orchestration/
  state.py          — ExperimentNode tree: create, traverse, query
  scheduler.py      — budget management, stage transitions
  accept_reject.py  — incumbent update logic
```

**ExperimentTree** (in state.py) manages the node graph:
- `add_root(plan) -> ExperimentNode`
- `add_child(parent_id, plan, edge_label) -> ExperimentNode`
- `get_incumbent() -> Optional[ExperimentNode]`
- `get_leaves() -> List[ExperimentNode]`
- `get_path_to_root(node_id) -> List[ExperimentNode]`
- `serialize() -> Dict` (preserves graph structure for future Graph RAG)

**Scheduler** controls the session flow:
- `should_start_optimization() -> bool` — all candidates have min_runs valid results
- `should_stop() -> bool` — budget exhausted or plateau (no improvement for N runs)
- `select_next_node() -> ExperimentNode` — round-robin in warmup, incumbent in optimize
- `get_stage() -> str` — "ideation", "warmup", "optimize"

**AcceptReject** updates the incumbent:
- `evaluate(parent: ExperimentNode, child: ExperimentNode) -> bool`
- Direction-aware comparison (higher-is-better vs lower-is-better)
- If accepted: child becomes new incumbent
- If rejected: child is marked, tree branch may be pruned later

### 4.3 Execution Layer

```
src/execution/
  autogluon_runner.py   — AutoGluon executor
  config_mapper.py      — RunConfig -> AutoGluon kwargs
  result_parser.py      — AutoGluon output -> RunResult + diagnostics
```

**Executor protocol:**
```
run(config: RunConfig, data_path: str, out_dir: str) -> RunResult
```

**ConfigMapper** translates agent decisions to AutoGluon API:
```
RunConfig.eval_metric      -> eval_metric="roc_auc"
RunConfig.model_families   -> hyperparameters={"GBM": {}, "XGB": {}}
RunConfig.presets           -> presets="medium_quality"
RunConfig.time_limit        -> time_limit=300
RunConfig.validation_policy -> num_bag_folds=5 or holdout_frac=0.2
RunConfig.feature_policy    -> included_columns / excluded_columns pre-filter
RunConfig.use_fit_extra     -> call fit_extra() on existing predictor
```

**ResultParser** extracts diagnostics from AutoGluon output:
- Calls `predictor.leaderboard()` for model rankings
- Calls `predictor.info()` for full metadata
- Computes overfitting gap (train vs val score if available)
- Extracts feature importances from best model
- Computes delta from parent node's metric
- Detects failure modes (timeout, all models failed, etc.)

**Optuna executor** added in Phase 4 as an alternative backend.
Agent chooses AutoGluon vs Optuna based on the experiment's goal.

### 4.4 Memory Layer

```
src/memory/
  run_store.py        — session experiment journal
  case_store.py       — cross-session knowledge base
  distiller.py        — RunStore -> CaseEntry at session end
  retrieval.py        — find similar cases for new tasks
  context_builder.py  — assembles SearchContext for agent
```

**RunStore** — append-only journal for the current session:
- `append(entry: RunEntry)`
- `get_history() -> List[RunEntry]`
- `get_incumbent() -> Optional[RunEntry]`
- `get_failed() -> List[RunEntry]`
- Storage: JSONL file under `experiments/{session}/decisions.jsonl`

**CaseStore** — persistent cross-session knowledge:
- `add(case: CaseEntry)`
- `search(task: TaskSpec, top_k: int) -> List[CaseEntry]`
- Storage: JSONL file, path configurable in project.yaml
- Retrieval v1: cosine similarity on task traits (simple)
- Retrieval v2 (future): vector + BM25 + reranking (from user's RAG patterns)
- Retrieval v3 (future): Graph RAG over ExperimentNode trees

**Distiller** — LLM-assisted session summarization:
- Input: full RunStore history + ExperimentTree
- Output: CaseEntry with distilled what_worked/what_failed/trajectory
- Runs once at session end
- Preserves tree_summary with edge labels for Graph RAG compatibility

**ContextBuilder** — assembles the SearchContext:
- Combines TaskSpec + DataProfile + RunStore history + CaseStore retrieval
- Adds tree metadata (current node, depth, branches)
- Adds budget info from Scheduler
- This is the single function that determines what the agent sees

---

## 5. LLM Abstraction

```
src/llm/
  backend.py          — LLMBackend protocol + create_backend() factory
  providers/
    anthropic.py      — Claude implementation
    openai.py         — GPT-4 implementation
```

No LangChain, no LlamaIndex. Thin protocol:

```
class LLMBackend(Protocol):
    def complete(self, messages: list[dict], temperature: float = 0.7,
                 response_format: Optional[dict] = None) -> str: ...
```

Each agent module receives `LLMBackend` as a dependency injection.
Provider selection configured in `project.yaml`.

---

## 6. Folder Structure

```
hybrid-agentic-ml/
|
+-- configs/
|   +-- project.yaml                 # task spec, LLM provider, data path
|   +-- search.yaml                  # budgets, thresholds, min_warmup_runs
|   +-- models.yaml                  # allowed model families + constraints
|   +-- seed_ideas.json              # optional bootstrapping hints
|
+-- prompts/
|   +-- ideator.md                   # data profiling + hypothesis generation
|   +-- selector.md                  # hypothesis -> ExperimentPlan
|   +-- refiner.md                   # results -> config changes
|   +-- reviewer.md                  # run quality assessment
|   +-- distiller.md                 # session -> CaseEntry summarization
|
+-- src/
|   +-- agents/
|   |   +-- manager.py               # ExperimentManager
|   |   +-- ideator.py               # data profile + case retrieval -> hypotheses
|   |   +-- selector.py              # hypothesis -> RunConfig
|   |   +-- refiner.py               # results -> config changes
|   |   +-- reviewer.py              # run quality assessment
|   |
|   +-- orchestration/
|   |   +-- state.py                 # ExperimentNode, ExperimentTree
|   |   +-- scheduler.py             # budget, stage transitions
|   |   +-- accept_reject.py         # incumbent update logic
|   |
|   +-- execution/
|   |   +-- autogluon_runner.py      # AutoGluon executor
|   |   +-- config_mapper.py         # RunConfig -> AutoGluon kwargs
|   |   +-- result_parser.py         # AutoGluon output -> RunResult + diagnostics
|   |
|   +-- memory/
|   |   +-- run_store.py             # session journal (RunEntry with diagnostics)
|   |   +-- case_store.py            # cross-session knowledge base (CaseEntry)
|   |   +-- distiller.py             # RunStore -> CaseEntry
|   |   +-- retrieval.py             # find similar cases
|   |   +-- context_builder.py       # assembles SearchContext
|   |
|   +-- llm/
|   |   +-- backend.py               # LLMBackend protocol + factory
|   |   +-- providers/
|   |       +-- anthropic.py
|   |       +-- openai.py
|   |
|   +-- models/
|       +-- task.py                   # TaskSpec, RunConfig, ExperimentPlan
|       +-- results.py               # RunResult, RunEntry, DataProfile, ModelEntry
|       +-- nodes.py                 # ExperimentNode, CaseEntry, SearchContext
|
+-- experiments/                     # timestamped session outputs
|   +-- {date}_{task_name}/
|       +-- manifest.json            # session metadata
|       +-- data_profile.json        # DataProfile for this session
|       +-- initial_hypotheses.json  # ideator output
|       +-- decisions.jsonl          # RunEntry journal
|       +-- tree.json                # serialized ExperimentTree (graph structure)
|       +-- runs/
|           +-- run_XXXX/
|               +-- config.json
|               +-- metrics.json
|               +-- predictor/       # saved AutoGluon predictor
|               +-- artifacts/       # plots, feature importances
|
+-- notebooks/
|   +-- 01_inspect_run.ipynb         # load run, see what agent saw
|   +-- 02_replay_decision.ipynb     # replay agent reasoning
|
+-- main.py                          # entrypoint
+-- requirements.txt
+-- .env.example
```

---

## 7. Session Flow (end to end)

```
1. User runs: python main.py --config configs/project.yaml

2. SETUP
   - Load TaskSpec from project.yaml
   - Initialize LLMBackend from config
   - Initialize empty RunStore, load CaseStore
   - Initialize ExperimentTree (empty)
   - Initialize Scheduler with budget from search.yaml

3. IDEATION
   - Ideator profiles dataset -> DataProfile
   - Ideator queries CaseStore for similar cases
   - Ideator proposes K hypotheses (default 3)
   - Each hypothesis -> root ExperimentNode in tree

4. WARM-UP LOOP (for each candidate)
   - Manager routes to Selector
   - Selector turns hypothesis into ExperimentPlan
   - ConfigMapper produces RunConfig
   - AutoGluon executor runs, returns RunResult
   - ResultParser enriches with diagnostics
   - RunEntry stored in RunStore
   - Reviewer assesses quality
   - Scheduler checks: all candidates have min_runs?
     - No -> next candidate (round-robin)
     - Yes -> transition to optimization

5. OPTIMIZATION LOOP (incumbent only)
   - ContextBuilder assembles SearchContext
   - Manager routes to Refiner
   - Refiner proposes config changes (new ExperimentPlan)
   - New ExperimentNode added as child of incumbent
   - Execute, parse, diagnose, store
   - AcceptReject: improved? -> new incumbent. Not? -> reject, stay.
   - Scheduler checks: budget left? plateau?
     - Continue -> next iteration
     - Stop -> exit loop

6. DISTILLATION
   - Distiller summarizes session: RunStore + ExperimentTree -> CaseEntry
   - CaseEntry includes tree_summary with edge labels
   - CaseEntry stored in CaseStore

7. OUTPUT
   - Print final incumbent metrics
   - Print experiment tree visualization
   - Save tree.json (Graph RAG compatible)
   - Save session manifest
```

---

## 8. Build Phases (breadth-first)

### Phase 1: Thin slice through all 4 layers
- `src/models/` — all data objects as Pydantic models
- `src/llm/` — backend protocol + one provider
- `src/execution/` — AutoGluon runner (tabular only), config mapper, basic result parser
- `src/memory/run_store.py` — append-only JSONL
- `src/orchestration/state.py` — ExperimentNode + basic tree
- `src/agents/manager.py` + `selector.py` — minimal decision loop
- `main.py` — wires all layers, runs 3-5 iterations on a demo dataset
- Goal: see the full loop execute end to end

### Phase 2: Deepen search + memory
- `src/orchestration/scheduler.py` — warmup/optimize transitions, budget tracking
- `src/orchestration/accept_reject.py` — incumbent update with direction-aware comparison
- `src/agents/ideator.py` — data profiling + case retrieval + hypothesis generation
- `src/memory/case_store.py` + `retrieval.py` — JSONL + cosine similarity retrieval
- `src/memory/distiller.py` — session -> CaseEntry summarization
- `src/memory/context_builder.py` — full SearchContext assembly
- Goal: the agent makes informed, context-grounded decisions

### Phase 3: Deepen agent reasoning
- `src/agents/refiner.py` — config refinement grounded in diagnostics
- `src/agents/reviewer.py` — run quality assessment (overfitting, leakage, suspicious patterns)
- Richer `result_parser.py` — feature importances, overfitting gap, parent comparison
- Full ExperimentTree serialization with edge labels
- Notebooks for debugging
- Goal: the agent reasons well about what to change and why

### Phase 4: Add Optuna + extensibility
- Optuna executor adapter (`src/execution/optuna_runner.py`)
- Agent chooses AutoGluon vs Optuna based on experiment goal
- seed_ideas.json support
- Second LLM provider
- Goal: the system has two execution backends and richer experiment variety

### Phase 5: Advanced (optional)
- Limited code editing (preprocessing/feature engineering modules only)
- Reporting module (markdown/LaTeX experiment summaries)
- Upgrade CaseStore to vector + BM25 + reranking (from user's RAG pipeline patterns)
- Graph RAG over ExperimentNode trees for sophisticated case retrieval
- BFTS-style progressive tree search (from AI-Scientist v2)
- Time series support via TimeSeriesPredictor

---

## 9. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Agent outputs structured JSON, not code | Higher success rate (AI-Scientist v1 lesson), matches AutoGluon's config-driven API |
| ExperimentNode tree, not flat list | Preserves lineage and rationale; future Graph RAG compatible |
| AutoGluon as default executor | Handles inner-loop optimization (model selection, ensembling, HPO) so agent focuses on problem framing |
| RunStore + CaseStore split | Short-term session memory vs long-term knowledge; different query patterns |
| Diagnostics stored in RunEntry | Analysis is part of memory, not a separate concern; agent needs it for reasoning |
| Thin LLM abstraction (no LangChain) | Multi-provider flexibility without framework lock-in; project is about ML architecture, not LLM tooling |
| Breadth-first build order | Working loop fast, deepen iteratively; matches learning goal |
| No top-level ideas/ folder | Hypotheses are per-session, stored in experiments/{session}/; seeds in configs/ |
| Case bank path in config, not top-level folder | case_store.py owns its storage; no confusing duplication |
| Edge labels on ExperimentNode | Explicit change descriptions enable future graph queries like "all experiments where switching metric helped" |

---

## 10. Reference Repos and What Was Borrowed

| Source | What was borrowed | What was NOT borrowed |
|--------|------------------|----------------------|
| AI-Scientist v1 | Template discipline, stage-based pipeline, filesystem state, seed ideas | LaTeX output, Aider code editing, Semantic Scholar novelty check |
| AI-Scientist v2 | AgentManager pattern, ExperimentNode tree, stage transitions, BFTS concepts | Full progressive tree search (deferred to Phase 5), free-form code generation |
| AutoGluon | Execution substrate, Predictor API, leaderboard/info for diagnostics, fit_extra() for incremental | Internal trainer/learner/model hierarchy (used as black box) |
| Optuna | Study/Trial concepts for Phase 4, Ask-Tell interface pattern, pruning ideas | Direct integration (deferred), sampler internals |
| User's RAG Pipeline | Hybrid retrieval pattern for CaseStore (Phase 5), reranking, confidence-based routing | Document chunking, chat memory, PubMed integration |

---

## 11. What the Agent Actually Controls (vs what AutoGluon controls)

```
AGENT DECIDES (outer loop):              AUTOGLUON DECIDES (inner loop):
- Which eval metric to optimize          - Which individual models to train
- Which model families are allowed       - Hyperparameter values per model
- Validation strategy (holdout/kfold)    - Early stopping criteria
- Feature inclusion/exclusion            - Bagging splits and weights
- Time budget allocation                 - Stacking levels and architecture
- Presets (quality tier)                 - Ensemble selection and weights
- Whether result is acceptable           - Model training order
- Whether to stop or continue            - Prediction calibration
- Problem framing and hypothesis         - Feature type inference
```

This separation is the key architectural insight: AutoML shrinks the fragile
part of the problem so the agent can focus on the semantic part.
