# Hybrid Agentic ML Framework

A self-improving ML experiment loop where an LLM agent reasons over **problem framing** while AutoGluon handles **inner-loop optimization** — connected by a graph-structured experiment tree that tracks every decision, rationale, and outcome across sessions.

Built as a learning exercise to deeply understand agentic ML architecture by studying [AI-Scientist v1/v2](https://github.com/SakanaAI/AI-Scientist), [AutoGluon](https://github.com/autogluon/autogluon), and [Optuna](https://github.com/optuna/optuna) — then building something coherent from first principles.

---

## What This Experiments With

### The core hypothesis

AutoML (AutoGluon) already solves the inner loop — model selection, ensembling, hyperparameter search. The interesting open question is: **can an LLM agent usefully operate above AutoML**, reasoning about the outer loop?

The agent controls things AutoGluon cannot decide for itself:
- Which evaluation metric to optimize (roc_auc vs f1_macro vs rmse?)
- Which model families to allow (are neural nets even appropriate here?)
- What validation strategy to use (holdout vs k-fold vs stratified?)
- Which features to include or exclude (suspected leakage? high cardinality noise?)
- How much time budget to allocate per iteration
- Whether a result is genuinely better or just noise

AutoGluon handles everything below that: individual model training, bagging, stacking, calibration, early stopping.

### Four-layer architecture

```
+-----------------------------------------------------------+
|  AGENT LAYER          — decides WHAT to try next          |
|  ideator, manager, selector, refiner, reviewer            |
+-----------------------------------------------------------+
|  ORCHESTRATION LAYER  — controls HOW experiments branch   |
|  ExperimentNode tree, scheduler, accept/reject gating     |
+-----------------------------------------------------------+
|  EXECUTION LAYER      — runs ML (boring and reliable)     |
|  AutoGluon runner, config mapper, result parser           |
+-----------------------------------------------------------+
|  MEMORY LAYER         — stores what happened              |
|  RunStore (session), CaseStore (cross-session), distiller |
+-----------------------------------------------------------+
```

### What makes it interesting to study

**Graph-structured lineage** — every experiment is a node in a tree. Edges carry labels describing what changed (`"switched to f1_macro due to class imbalance"`, `"dropped leaky columns, added k-fold"`). This is not just a log — it's a queryable graph. Designed for future Graph RAG.

**Two-stage search** — warm-up phase explores 3 diverse hypotheses in parallel. Optimization phase deepens the best one. An accept/reject gate prevents the agent from making things worse.

**Cross-session knowledge** — at session end, a distiller LLM summarizes what worked, what failed, and why into a `CaseEntry`. Future sessions retrieve similar cases and start with informed priors instead of blind search.

**No LangChain** — thin LLM abstraction, no framework lock-in. The interesting architecture is the ML experiment loop, not the LLM tooling.

---

## Architecture at a Glance

```
Task + Data
  ↓
IDEATION  (data profiling → hypotheses → root ExperimentNodes)
  ↓
WARM-UP   (3 candidates, each becomes a root node in the tree)
  ↓
OPTIMIZE  (deepen the best candidate, accept/reject each step)
  ↓
DISTILL   (session → CaseEntry → CaseStore for future retrieval)
```

Each iteration: `ExperimentPlan (JSON)` → `RunConfig` → `AutoGluon` → `RunResult + diagnostics` → `RunEntry (stored)` → agent review → next decision.

The agent never writes code. It outputs structured `ExperimentPlan` JSON — which metric, which models, which validation strategy, which features. AutoGluon executes. The loop runs.

---

## Project Structure

```
src/
  agents/          — LLM agents (manager, ideator, selector, refiner, reviewer)
  orchestration/   — ExperimentNode tree, scheduler, accept/reject
  execution/       — AutoGluon runner, config mapper, result parser
  memory/          — RunStore, CaseStore, distiller, context builder
  models/          — Pydantic v2 data models (TaskSpec, ExperimentPlan, RunEntry, ...)
  llm/             — thin LLM backend (Anthropic + OpenAI, no framework)

configs/           — task spec, search config, allowed models, seed ideas
prompts/           — agent prompt templates (markdown, loaded at runtime)
experiments/       — timestamped session outputs, run artifacts, tree.json
docs/
  plans/           — architecture design + implementation plans
```

---

## Roadmap

### Phase 1 — Thin slice through all 4 layers ✅
- All Pydantic v2 data models
- LLM backend with Anthropic + OpenAI providers
- AutoGluon execution + result parser
- RunStore (append-only JSONL session journal)
- ExperimentNode tree with edge labels
- SelectorAgent (hypothesis → ExperimentPlan via LLM)
- ExperimentManager (warmup/optimize routing)
- Full session loop wired end-to-end
- 50 tests, all green

### Phase 2 — Deepen search + memory
- `IdeatorAgent` — data profiling + CaseStore retrieval → initial hypotheses
- `CaseStore` — cross-session JSONL knowledge base
- `Distiller` — LLM summarizes session into CaseEntry at end
- `ContextBuilder` — assembles the SearchContext briefing for each agent call
- `RetrievalModule` — cosine similarity on task traits (v1), hybrid RAG (Phase 5)
- Goal: agent makes context-grounded decisions, not blind search

### Phase 3 — Deepen agent reasoning
- `RefinerAgent` — config refinement grounded in diagnostics and parent comparison
- `ReviewerAgent` — flags overfitting, suspected leakage, suspicious patterns
- Richer `ResultParser` — feature importances, overfitting gap, parent delta
- Full ExperimentTree serialization with edge labels preserved
- Jupyter notebooks for inspecting runs and replaying agent decisions

### Phase 4 — Add Optuna + extensibility
- Optuna executor adapter as second execution backend
- Agent chooses AutoGluon vs Optuna based on experiment goal
- seed_ideas.json bootstrapping support
- Goal: two execution backends, richer experiment variety

### Phase 5 — Advanced (optional)
- Upgrade CaseStore to vector + BM25 + reranking (from prior RAG pipeline work)
- Graph RAG over ExperimentNode trees — query by edge labels, not just traits
- BFTS-style progressive tree search (inspired by AI-Scientist v2)
- Reporting module (markdown summaries per session)
- Time series support via AutoGluon TimeSeriesPredictor

---

## Reference Repos Studied

| Repo | What was borrowed |
|------|-------------------|
| [SakanaAI/AI-Scientist v1](https://github.com/SakanaAI/AI-Scientist) | Template discipline, stage-based pipeline, seed ideas pattern |
| [SakanaAI/AI-Scientist v2](https://github.com/SakanaAI/AI-Scientist-v2) | AgentManager pattern, ExperimentNode tree, BFTS concepts |
| [autogluon/autogluon](https://github.com/autogluon/autogluon) | Execution substrate, Predictor API, leaderboard/info for diagnostics |
| [optuna/optuna](https://github.com/optuna/optuna) | Study/Trial concepts, Ask-Tell interface (Phase 4) |

---

## Quick Start

```bash
# 1. Clone and set up environment
git clone <repo>
cd agentic-ml
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install autogluon.tabular  # takes 2–3 minutes

# 2. Add your API key
cp .env.example .env
# edit .env: OPENAI_API_KEY=sk-... or ANTHROPIC_API_KEY=sk-ant-...

# 3. Configure provider in configs/project.yaml
# llm:
#   provider: "openai"       # or "anthropic"
#   model: "gpt-4o"

# 4. Run tests
python3 -m pytest tests/ -q

# 5. Run the experiment loop on demo Titanic data
python3 main.py
```

Session outputs land in `experiments/{date}_{task}/` — including `decisions.jsonl` (full RunEntry journal) and `tree.json` (serialized ExperimentNode graph).

---

## Tech Stack

- **Python 3.12**, **Pydantic v2** (all data models)
- **AutoGluon Tabular** (ML execution backend)
- **Anthropic SDK** + **OpenAI SDK** (thin LLM abstraction, no LangChain)
- **pytest** (TDD throughout, 50+ tests)
- **pandas**, **scikit-learn**, **PyYAML**
