# Hybrid Agentic ML Framework

A self-improving ML experiment loop where LLM agents reason over **problem framing**, **preprocessing**, and **feature engineering** while AutoGluon handles **inner-loop optimization** — connected by a graph-structured experiment tree that tracks every decision, rationale, and outcome across sessions.

Built as a learning exercise to deeply understand agentic ML architecture by studying [AI-Scientist v1/v2](https://github.com/SakanaAI/AI-Scientist), [AutoGluon](https://github.com/autogluon/autogluon), and [Optuna](https://github.com/optuna/optuna) — then building something coherent from first principles.

---

## What This Does

### The core hypothesis

AutoML (AutoGluon) already solves the inner loop — model selection, ensembling, hyperparameter search. The interesting open question is: **can LLM agents usefully operate above AutoML**, reasoning about the outer loop?

The agents control things AutoGluon cannot decide for itself:
- Which evaluation metric to optimize (roc_auc vs f1_macro vs rmse?)
- Which model families to allow (are neural nets even appropriate here?)
- What validation strategy to use (holdout vs k-fold vs stratified?)
- Which features to engineer, transform, or drop — with leakage defense by construction
- How to preprocess raw data (missing values, encoding, scaling)
- How much time budget to allocate per iteration
- Whether a result is genuinely better or just noise

AutoGluon handles everything below that: individual model training, bagging, stacking, calibration, early stopping.

### Two campaign paths

The system supports two independent campaign types that share the core training spine:

**Preprocessing Campaign** — `CampaignOrchestrator` + `PreprocessingAgent` generates and validates data cleaning code via a ReAct loop with subprocess-isolated validation.

**Feature Engineering Campaign** — `FeatureCampaignOrchestrator` + `FeatureEngineeringAgent` proposes, audits, and executes predictive features for ecommerce lifecycle tasks (churn, repurchase, LTV) using bounded templates + DSL with mandatory leakage auditing.

---

## Architecture

### Six-layer design

```
+-----------------------------------------------------------+
|  AGENT LAYER          — decides WHAT to try next          |
|  IdeatorAgent, SelectorAgent, RefinerAgent, Manager       |
|  PreprocessingAgent: ReAct inspect/generate loop          |
|  FeatureEngineeringAgent: decision → audit → execute      |
+-----------------------------------------------------------+
|  ORCHESTRATION LAYER  — controls HOW experiments branch   |
|  CampaignOrchestrator: preprocessing campaigns            |
|  FeatureCampaignOrchestrator: feature eng campaigns       |
|  ExperimentTree, Scheduler, AcceptReject gating           |
+-----------------------------------------------------------+
|  FEATURE ENGINEERING LAYER — bounded feature execution    |
|  TemplateRegistry: 20 named ecommerce feature templates   |
|  BoundedExecutor: dispatches DSL configs to templates     |
|  FeatureValidator: row count, target, null/constant       |
|  DSL: 14-operator surface with time-op leakage guards     |
+-----------------------------------------------------------+
|  EXECUTION LAYER      — runs ML (boring and reliable)     |
|  AutoGluon runner, ConfigMapper, ResultParser             |
|  PreprocessingExecutor, ValidationHarness (subprocess)    |
+-----------------------------------------------------------+
|  MEMORY LAYER         — stores what happened              |
|  RunStore: session journal (JSONL)                        |
|  FeatureHistoryStore: empirical feature experiment memory  |
|  ContextBuilder, FeatureContextBuilder: agent briefings   |
+-----------------------------------------------------------+
|  DATA LAYER           — reference knowledge               |
|  Static reference folders (ecommerce features, leakage)   |
|  Prompt templates, seed cases, task specs                 |
+-----------------------------------------------------------+
```

### Feature engineering pipeline (Phase 5)

```
Context Assembly → LLM Decision Call → Leakage Audit Call → Bounded Execution → Validation
                   FeatureDecision      FeatureAuditVerdict   BoundedExecutor    FeatureValidator
                   (JSON contract)      (mandatory, 5 checks) (14 DSL operators) (row/target/null)
```

- **20 template functions**: RFM (recency, frequency, monetary), order (AOV, basket size, category diversity), temporal (windowed count/sum/mean/nunique, days_since), transforms (log1p, clip, bucketize, is_missing), composites (safe_divide, subtract, add, multiply, ratio_to_baseline)
- **Leakage defense by construction**: time-based features require entity_key, time_col, and explicit window — enforced at DSL validation before execution
- **Codegen escape hatch** (Phase 2): explicit escalation path with subprocess sandbox and separate guardrail audit

### What makes it interesting to study

**Graph-structured lineage** — every experiment is a node in a tree. Edges carry labels describing what changed. Designed for future Graph RAG.

**Two-stage search** — warm-up phase explores 3 diverse hypotheses in parallel. Optimization phase deepens the best one. An accept/reject gate prevents the agent from making things worse.

**Empirical feature memory** — `FeatureHistoryStore` records every feature proposal with before/after metrics and distilled takeaways. Future iterations learn from past experiments, not just prompts.

**Mandatory leakage auditing** — every feature proposal passes through an LLM leakage audit (5 checks: target usage, future timestamps, post-outcome joins, unbounded aggregations, missing entity boundaries) before execution is allowed.

**No LangChain** — thin LLM abstraction, no framework lock-in. The interesting architecture is the ML experiment loop, not the LLM tooling.

---

## Project Structure

```
src/
  agents/          — LLM agents (manager, ideator, selector, refiner, reviewer,
                     preprocessing_agent, feature_engineer)
  orchestration/   — CampaignOrchestrator, FeatureCampaignOrchestrator,
                     ExperimentNode tree, scheduler
  execution/       — AutoGluon runner, config mapper, result parser,
                     preprocessing executor, validation harness
  features/        — template registry, bounded executor, DSL validation,
                     feature validator, history store, context builder
    templates/     — customer (RFM), order, temporal, transforms, composites
  memory/          — RunStore, FeatureHistoryStore, context builders
  models/          — Pydantic v2 data models (TaskSpec, FeatureDecision,
                     FeatureSpec variants, CampaignResult, ...)
  llm/             — thin LLM backend (Anthropic + OpenAI, no framework)

configs/           — task spec, search config, allowed models, seed ideas
prompts/
  feature_engineering/ — decision, leakage audit, codegen prompts
references/
  feature_engineering/ — ecommerce feature knowledge, leakage patterns
experiments/       — timestamped session outputs, campaign artifacts
docs/
  architecture/    — current system state (source of truth)
  changes/         — implementation log
  plans/           — design docs and implementation plans
  decisions/       — architecture decision records
```

---

## Roadmap

### Phase 1 — Thin slice through all 4 layers ✅
- All Pydantic v2 data models, LLM backend, AutoGluon execution
- RunStore, ExperimentNode tree, SelectorAgent, ExperimentManager
- Full session loop wired end-to-end

### Phase 2 — Deepen search + memory ✅
- IdeatorAgent, CaseStore, Distiller, ContextBuilder, RetrievalModule
- Agent makes context-grounded decisions, not blind search

### Phase 3 — Deepen agent reasoning ✅
- RefinerAgent, ReviewerAgent, richer ResultParser
- Full ExperimentTree serialization with edge labels

### Phase 4 — Preprocessing + Campaign orchestration ✅
- CampaignOrchestrator with plateau detection and budget stops
- PreprocessingAgent with ReAct loop and subprocess-isolated validation
- Semantic seed bank for preprocessing knowledge

### Phase 5 — Feature engineering subsystem ✅
- FeatureEngineeringAgent: decision → leakage audit → bounded execution
- 20 ecommerce feature templates (RFM, order, temporal, transforms, composites)
- 14-operator DSL with time-op leakage guards
- FeatureCampaignOrchestrator: baseline → feature iterations → retrain
- FeatureHistoryStore: empirical experiment memory (JSONL)
- Static reference folders replace RAG/vector retrieval
- Mothballed CaseStore, PreprocessingStore, EmbeddingRetriever

### Phase 5b — Codegen escape hatch (next)
- CodegenSandbox: subprocess-isolated code execution
- Codegen generation + guardrail audit prompts
- Wired into FeatureEngineeringAgent as explicit escalation

### Future
- Graph RAG over ExperimentNode trees
- BFTS-style progressive tree search (AI-Scientist v2)
- Per-row cutoff semantics for temporal features (adversarial review finding)
- Time series support via AutoGluon TimeSeriesPredictor

---

## Reference Repos Studied

| Repo | What was borrowed |
|------|-------------------|
| [SakanaAI/AI-Scientist v1](https://github.com/SakanaAI/AI-Scientist) | Template discipline, stage-based pipeline, seed ideas pattern |
| [SakanaAI/AI-Scientist v2](https://github.com/SakanaAI/AI-Scientist-v2) | AgentManager pattern, ExperimentNode tree, BFTS concepts |
| [autogluon/autogluon](https://github.com/autogluon/autogluon) | Execution substrate, Predictor API, leaderboard/info for diagnostics |
| [optuna/optuna](https://github.com/optuna/optuna) | Study/Trial concepts, Ask-Tell interface |

---

## Quick Start

```bash
# 1. Clone and set up environment
git clone <repo>
cd agentic-ml
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install autogluon.tabular  # takes 2-3 minutes

# 2. Add your API key
cp .env.example .env
# edit .env: OPENAI_API_KEY=sk-... or ANTHROPIC_API_KEY=sk-ant-...

# 3. Run tests (253 tests)
python3 -m pytest tests/ -q

# 4. Run the experiment loop on demo data
python3 main.py
```

Session outputs land in `experiments/{date}_{task}/` — including `decisions.jsonl` (full RunEntry journal) and `tree.json` (serialized ExperimentNode graph).

---

## Tech Stack

- **Python 3.12**, **Pydantic v2** (all data models)
- **AutoGluon Tabular** (ML execution backend)
- **Anthropic SDK** + **OpenAI SDK** (thin LLM abstraction, no LangChain)
- **pandas**, **numpy** (feature engineering templates)
- **pytest** (TDD throughout, 253 tests)
