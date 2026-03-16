## My initial thinking of this Agentic ML Project 

their “agent loop” is not really a swarm of many independent agents.
It is much closer to a planner-centered, self-improving ML workflow that repeatedly does four things: choose a model direction, edit a constrained code scaffold, run it, read the results, and use the logs as memory for the next decision. The paper explicitly frames TS-Agent as a modular iterative decision process over model selection, code refinement, and fine-tuning, driven by a planner agent, contextual memory, and experimental feedback.

Here is the architecture, upside down.

The loop starts with three ingredients:
a task description 
𝑤
ℎ
𝑎
𝑡
𝑝
𝑟
𝑜
𝑏
𝑙
𝑒
𝑚
𝑡
𝑜
𝑠
𝑜
𝑙
𝑣
𝑒
𝑎
𝑛
𝑑
ℎ
𝑜
𝑤
𝑡
𝑜
𝑒
𝑣
𝑎
𝑙
𝑢
𝑎
𝑡
𝑒
𝑖
𝑡
whatproblemtosolveandhowtoevaluateit, a scaffolded executable template 
𝑡
ℎ
𝑒
𝑖
𝑟
‘
𝑡
𝑟
𝑎
𝑖
𝑛
.
𝑝
𝑦
‘
their‘train.py‘, and three external knowledge sources: Case Bank 
𝑝
𝑎
𝑠
𝑡
𝑠
𝑜
𝑙
𝑣
𝑒
𝑑
𝑡
𝑎
𝑠
𝑘
𝑠
/
𝑝
𝑙
𝑎
𝑦
𝑏
𝑜
𝑜
𝑘
𝑠
pastsolvedtasks/playbooks, Code Base 
𝑖
𝑚
𝑝
𝑙
𝑒
𝑚
𝑒
𝑛
𝑡
𝑒
𝑑
𝑚
𝑜
𝑑
𝑒
𝑙
𝑠
+
𝑒
𝑣
𝑎
𝑙
𝑢
𝑎
𝑡
𝑖
𝑜
𝑛
𝑚
𝑒
𝑎
𝑠
𝑢
𝑟
𝑒
𝑠
implementedmodels+evaluationmeasures, and Refinement Knowledge Bank 
𝑡
𝑖
𝑝
𝑠
𝑓
𝑜
𝑟
𝑝
𝑟
𝑒
𝑝
𝑟
𝑜
𝑐
𝑒
𝑠
𝑠
𝑖
𝑛
𝑔
,
𝑡
𝑟
𝑎
𝑖
𝑛
𝑖
𝑛
𝑔
𝑜
𝑝
𝑡
𝑖
𝑚
𝑖
𝑧
𝑎
𝑡
𝑖
𝑜
𝑛
,
ℎ
𝑦
𝑝
𝑒
𝑟
𝑝
𝑎
𝑟
𝑎
𝑚
𝑒
𝑡
𝑒
𝑟
𝑡
𝑢
𝑛
𝑖
𝑛
𝑔
,
𝑎
𝑛
𝑑
𝑒
𝑣
𝑎
𝑙
𝑢
𝑎
𝑡
𝑖
𝑜
𝑛
tipsforpreprocessing,trainingoptimization,hyperparametertuning,andevaluation. The goal is not to invent everything from scratch, but to fill and update the scaffold so it becomes executable, trains successfully, and improves the target metric.

So the real loop is:

Task → retrieve relevant prior cases → shortlist model families → edit the scaffold in a controlled way → run experiment → log metrics + code changes + rationale → update memory/context → decide next edit.
That is the heart of it. The paper even formalizes the final code-editing decision as a composition of sub-decisions: choose model, choose hyperparameters, choose refinements.

The most important design choice is that they restrict the search space. Instead of asking an LLM to freely generate an entire ML pipeline, they make the agent operate over a modular code template. The action space is only four things:
A_model = select model/measure,
A_refinement = apply training/refinement edits,
A_fine-tune = update hyperparameters,
A_logging = execute and record what happened.
This matters a lot because it makes the system more controllable and auditable. The paper even notes that only the refinement action is allowed to introduce coding bugs, which localizes failure and makes debugging easier.

Their “memory” is also very concrete, not magical. At iteration 
𝑡
t, memory is basically the history of prior experiment logs and prior versions of the script. Context is then formed by combining that dynamic memory with the static external resources and the task itself. In plain English: the agent does not just remember text; it remembers which code version led to which metrics and which change helped or hurt. That is what gives the loop its self-improving behavior.

The paper’s feedback loop is therefore:

initialize context

do model selection, refinement, and fine-tuning

execute and log results

append results to memory

rebuild context

repeat until budget ends.

That is literally their algorithmic skeleton.

Now the two-stage design is the bigger idea behind the loop.

Stage 1: model pre-selection.
They do not optimize every possible model equally. They first use case-based reasoning over the Case Bank plus task description to retrieve similar past problems and narrow the search to top-
𝑘
k candidate models. So the loop begins with a smart prior, not blind search.

Stage 2: code refinement.
This is the true self-improvement loop. They run a two-phase round-robin search:
during Warm-up, each shortlisted candidate gets a few refinement/tuning cycles; then the best combination of model + refinements + hyperparameters is selected; during Optimization, only the winning candidate continues to receive more iterations. If a change improves loss, it becomes the new incumbent; if not, it is rejected and the system reverts. That “accept beneficial / reject harmful” rule is the core improvement mechanism.

So if I abstract away the time-series wording, the generic workflow is:

retrieve → route → instantiate → refine → tune → execute → evaluate → remember → repeat.

That is the reusable architecture you are actually interested in.

What makes it “self-improving” is also worth being precise about.
It is not self-improving in the sense of online gradient updates to the agent itself, and it is not training the LLM. The paper says the planner agent “updates its policy” based on experimental outcomes, but operationally this appears to mean decision updates through logged experiment feedback, contextual memory, and selective reuse of knowledge banks, not RL on model weights. That is my interpretation from how the system is described.

Another important insight: the architecture is powerful because it combines three kinds of prior at once.

First, case prior: “what worked on similar tasks before?”
Second, implementation prior: “what runnable code/modules already exist?”
Third, refinement prior: “what expert knobs and fixes should be tried when logs look a certain way?”
This is why it is much stronger than a plain LLM coder or a plain AutoML searcher. It is not searching from zero.

The paper’s empirical discussion also supports this interpretation: they attribute TS-Agent’s robustness partly to the fact that it refines existing models from a model bank instead of generating everything from scratch, which reduces variance across LLM backbones.

Now, connecting this to your second PDF:
my read from your screenshot deck is that the same logic has already been generalized beyond TS-Agent into a broader Fab-Agent / AI-box factory pattern. Visually, it mirrors the same structure: case bank + analytics code base + refinement/guardrails KB → task ticket / case retrieval → AI box selection → iterative refinement → results & logs → closed-loop learning. So the deeper connection is that TS-Agent seems to be the workflow primitive, while the deck shows the organizational / deployment wrapper around that primitive.

That means the truly reusable architecture is not “TS forecasting agent.”
It is:

a guarded closed-loop ML operating system for domain problems.

If you want to rebuild this for another domain later, the irreducible components you should preserve are:

a fixed task contract: what the input task spec must contain

a case bank: solved analogues / playbooks

a code bank: runnable modules, not just papers

a refinement bank: expert heuristics tied to failure patterns

a constrained executable scaffold: so edits are bounded

an experiment memory/log store: code diff + metrics + rationale

a two-stage search policy: shortlist first, then iterate deeply on the winner

That synthesis follows directly from the paper’s architecture and stage design.




## And then this is my friends recommandation (The recommandation is also referred to the reference repos from github, that can be used as a reference)

It is a student-project-friendly hybrid that borrows:

from AI Scientist v1: the strong template discipline (experiment.py, plot.py, prompt.json, seed_ideas.json, latex/template.tex);

from AI Scientist v2: the separate ideation step, bfts_config.yaml, experiment-manager/tree-search style, and timestamped experiments/.../logs/... outputs;

from AutoGluon: the idea that the outer API is task-level predictors, while the inner stack is roughly Predictor → Learner → Trainer → Models, with multiple candidate models trained/ensembled and stored under a run path.

I would not build your project as fully template-free code generation like AI Scientist v2, because that repo explicitly warns it executes LLM-written code and should be sandboxed, while v1’s template approach is much more stable for bounded experiments. AutoGluon also already gives you a strong execution layer for tabular and time-series tasks, so you can let the agent decide what to try without letting it freely author the whole training stack.

Recommended hybrid folder structure
hybrid-agentic-ml/
├── README.md
├── requirements.txt
├── .env.example
│
├── configs/
│   ├── project.yaml                 # global settings
│   ├── bfts_config.yaml             # borrowed idea from AI Scientist v2
│   ├── guardrails.yaml              # hard constraints / forbidden actions
│   ├── metrics.yaml                 # primary + secondary metrics
│   ├── search_spaces/
│   │   ├── tabular.yaml
│   │   └── timeseries.yaml
│   └── routing/
│       ├── modality_rules.yaml
│       └── shortlist_rules.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── schemas/
│       ├── tabular_schema.json
│       └── timeseries_schema.json
│
├── ideas/
│   ├── topic.md                     # v2-style topic description
│   ├── generated_ideas.json         # output of ideation
│   └── seed_ideas.json              # v1-style seeded examples
│
├── case_bank/
│   ├── cases.jsonl                  # distilled successful/failed patterns
│   ├── retrieval_index/             # optional embeddings / metadata index
│   └── templates/
│       ├── case_entry.schema.json
│       └── run_log.schema.json
│
├── prompts/
│   ├── ideation.md
│   ├── model_selector.md
│   ├── refinement_agent.md
│   ├── reviewer.md
│   └── summarizer.md
│
├── templates/
│   ├── tabular/
│   │   ├── experiment.py            # v1-style bounded scaffold
│   │   ├── plot.py
│   │   ├── prompt.json
│   │   └── latex/
│   │       └── template.tex
│   └── timeseries/
│       ├── experiment.py
│       ├── plot.py
│       ├── prompt.json
│       └── latex/
│           └── template.tex
│
├── src/
│   ├── agents/
│   │   ├── manager.py               # experiment manager, inspired by v2
│   │   ├── ideation.py
│   │   ├── selector.py
│   │   ├── refiner.py
│   │   ├── tuner.py
│   │   ├── reviewer.py
│   │   └── summarizer.py
│   │
│   ├── orchestration/
│   │   ├── state.py                 # run state, frontier, incumbent
│   │   ├── tree_search.py           # best-first / bounded search
│   │   ├── scheduler.py             # warm-up vs optimization budget
│   │   ├── accept_reject.py
│   │   └── checkpoints.py
│   │
│   ├── adapters/
│   │   ├── autogluon/
│   │   │   ├── predictor_factory.py
│   │   │   ├── tabular_runner.py
│   │   │   ├── timeseries_runner.py
│   │   │   ├── hyperparam_mapper.py
│   │   │   └── leaderboard_parser.py
│   │   └── sklearn/
│   │       ├── baseline_runner.py
│   │       └── metrics.py
│   │
│   ├── domain/
│   │   ├── task_contracts.py        # task signature / schema checks
│   │   ├── routing.py               # decide tabular vs timeseries
│   │   ├── feature_policies.py
│   │   └── validation_policies.py
│   │
│   ├── memory/
│   │   ├── case_store.py
│   │   ├── run_store.py
│   │   ├── retrieval.py
│   │   └── summarise_run.py
│   │
│   ├── execution/
│   │   ├── runner.py                # one experiment execution
│   │   ├── sandbox.py
│   │   ├── artifact_store.py
│   │   └── diff_utils.py
│   │
│   └── reporting/
│       ├── plot_builder.py
│       ├── markdown_report.py
│       └── latex_report.py
│
├── experiments/
│   └── 2026-03-16_customer-churn/
│       ├── run_manifest.json
│       ├── state.json
│       ├── logs/
│       │   ├── manager.log
│       │   ├── decisions.jsonl
│       │   └── unified_tree_viz.html
│       ├── runs/
│       │   ├── run_0001/
│       │   │   ├── task.json
│       │   │   ├── candidate.json
│       │   │   ├── config.json
│       │   │   ├── metrics.json
│       │   │   ├── artifacts/
│       │   │   └── summary.json
│       │   └── run_0002/
│       ├── best/
│       │   ├── incumbent.json
│       │   ├── predictor/           # saved AutoGluon predictor
│       │   └── plots/
│       └── report/
│           ├── summary.md
│           └── final.pdf
│
└── notebooks/
    ├── 01_data_profile.ipynb
    └── 02_debug_one_run.ipynb
Why this structure makes sense

The templates/ folder is directly inspired by AI Scientist v1, where a template is expected to contain experiment.py, plot.py, prompt.json, seed_ideas.json, and latex/template.tex. That is the right mental model for your project too: keep the executable surface bounded and inspectable.

The ideas/, bfts_config.yaml, and experiments/<timestamp>/logs/unified_tree_viz.html pieces come from AI Scientist v2. Its README separates idea generation from the main experiment run, uses a dedicated tree-search config file, and writes timestamped experiment folders with log artifacts including a tree visualization.

The adapters/autogluon/ layer is there because AutoGluon already organizes automation around task-level predictors like TabularPredictor and TimeSeriesPredictor, while its internals expose a learner and a trainer that control the training of multiple models. Its docs also show that you can specify candidate model families through a hyperparameters dictionary and that trained models live under a predictor path. That makes AutoGluon a very natural “execution substrate” under your agent loop.

The architectural split I’d use

I would split your system into four layers:

1. Agent layer
This decides what to do next: generate ideas, shortlist models, apply refinement, tune hyperparameters, review runs, and summarize lessons. This is your AI Scientist-inspired piece, especially the v2-style manager.

2. Search layer
This owns warm-up versus optimization, acceptance/rejection, frontier expansion, and debug depth. That is your simplified version of v2’s progressive tree-search idea.

3. Execution layer
This should be boring and reliable. For tabular/time-series projects, let AutoGluon do most of the real fitting, ensembling, and metric optimization instead of asking the LLM to rewrite training code every time. AutoGluon explicitly tunes hyperparameters, early stopping, and ensemble weights against the evaluation metric.

4. Memory layer
This stores cases, runs, diffs, metrics, and distilled “what worked / what failed.” That is the part both AI Scientist repos imply, but neither gives you as a clean reusable product layer, so you should make it explicit in your own build. The v1 template system encourages structured outputs per run, and v2’s experiment folders/logs give you the right storage shape.

How the flow should work

I’d wire the flow like this:

topic/task spec
  -> ideation
  -> shortlist candidates
  -> instantiate bounded experiment scaffold
  -> run AutoGluon / baseline executor
  -> collect metrics + artifacts
  -> summarize run
  -> accept/reject change
  -> update case bank + run logs
  -> next search step

That flow is basically v2’s staged search loop, but with v1-style bounded scaffolds and AutoGluon as the model-training backend instead of free-form code generation. That gives you far better controllability.

What each folder should really do

configs/
Put all search and policy knobs here, especially anything that should not be invented by the LLM at runtime. Mirror v2’s bfts_config.yaml idea for search settings like workers, steps, and debug depth.

ideas/
Keep both v2-style topic-driven idea generation and v1-style seed ideas. In practice, this is powerful: let the system brainstorm broadly, but still anchor it with a few seeded examples so it does not drift. v2 explicitly uses a markdown topic file for ideation, while v1 explicitly supports seed_ideas.json.

templates/
Keep one scaffold per task family, not per project. For example, templates/tabular/experiment.py should know how to train one bounded family of pipelines, but accept a config that tells it which models/hyperparameters/refinements to use. This keeps the system teachable and debuggable. That is the best lesson to borrow from v1.

adapters/autogluon/
This is where the agent’s decisions become concrete AutoGluon calls. For example:

route tabular tasks to TabularPredictor

route forecasting tasks to TimeSeriesPredictor

convert your candidate spec into AutoGluon hyperparameters

read leaderboard/model graph artifacts back into your memory layer.
That maps directly onto AutoGluon’s predictor APIs and model-hyperparameter dict style.

experiments/
Use v2’s timestamped run folder idea. Every experiment should have its own self-contained folder with manifest, decisions, raw run outputs, best incumbent, and final report. v2 explicitly stores runs under timestamped experiment folders and places tree/log outputs under logs/.

The one design choice I would change from AI Scientist

I would not let the agent freely edit experiment.py except in a very constrained way. Instead, I would let it edit:

candidate model set

validation policy

feature policy

AutoGluon hyperparameter dict

time budget

acceptance threshold

guardrails

That still gives you “agentic ML,” but the model is searching over a structured config space, not rewriting the training engine. This is safer, and it matches AutoGluon’s own API style much better. AutoGluon already expects structured configuration for labels, metrics, predictors, and model hyperparameters.

Minimal starter version

If you want the smallest useful version, build only this:

configs/
ideas/
case_bank/
templates/tabular/
src/agents/{manager,selector,refiner}.py
src/adapters/autogluon/tabular_runner.py
src/orchestration/{scheduler,accept_reject}.py
experiments/

And support only:

one task type: tabular classification

one execution engine: TabularPredictor

one warm-up stage: shortlist 3 candidates

one optimization stage: tune the winner

one memory store: cases.jsonl + decisions.jsonl

That is enough to learn the architecture without drowning in complexity. AutoGluon is particularly well-suited here because TabularPredictor.fit() is already designed to train strong tabular models quickly, and its trainer abstraction is explicitly about training/ensembling many models.

My blunt recommendation

For your project, I would build:

AI Scientist v1-style templates

AI Scientist v2-style manager/search/log folders

AutoGluon runners as the execution core

your own case bank and guardrails layer


The clean way to think about it is:

the search layer, the agent layer, AutoML, and Optuna are not the same thing.
They operate at different levels.

Search layer = outer-loop control over experiment branches.
Agent layer = reasoning/policy that decides the next action.
AutoML = inner-loop model training/search inside a predefined ML space.
Optuna = generic optimizer for any objective function you expose to it.

So there is no contradiction in combining them. The only mistake is letting two layers optimize the same knobs in an uncoordinated way.

1. What the search layer really is

When I said “search layer,” I did not mean “hyperparameter search” only.

I meant the outer experiment search policy: which branch to try next, when to go wider versus deeper, when to stop debugging a failed branch, when to abandon a model family, and when to spend more budget on the current best branch. That is exactly how AI Scientist-v2 frames its tree search: num_workers, steps, num_drafts, max_debug_depth, and debug_prob govern how many paths to explore, how long to explore them, and how much effort to spend debugging failures.

So yes: in a TS-Agent / AI-Scientist-like system, the search layer is the thing that decides:

explore new candidate or refine current best,

accept or reject a child run,

retry/debug or kill a failing branch,

switch from warm-up to optimization.

That is different from AutoGluon. AutoGluon is mostly the execution engine for a chosen experiment configuration. It trains many models, bags them, and stack-ensembles them; its docs explicitly say it does not primarily rely on HPO for strong performance, but on training diverse models plus bagging and stacking.

My recommendation is: for your project, the search layer can be much simpler than AI Scientist-v2. You do not need full progressive tree search at first. A two-stage policy is enough:

warm-up: try a few candidate experiment settings,

optimization: deepen only the best valid branch.

That is my design recommendation, not something AutoGluon gives you out of the box.

2. If AutoGluon does the ML, is there still anything meaningful left for the agent to do?

Yes — a lot.

AutoGluon can already optimize many inner-loop choices: it uses validation data for early stopping, hyperparameter tuning, and ensembling; it lets you choose model families via the hyperparameters dictionary; and it exposes knobs like presets, dynamic_stacking, calibrate_decision_threshold, tuning_data, and other fit-time controls. It also supports explicit HPO search spaces and hyperparameter_tune_kwargs when you want them.

So the LLM/agent should usually not micromanage every tree depth and learning rate itself. Instead, it should reason about the outer-loop decisions that AutoML is bad at inferring from context alone, such as:

are we optimizing the right metric,

is the validation split realistic,

should we force group-aware or time-aware validation,

which model families are allowed,

how much time budget to allocate,

whether interpretability or latency should override raw score,

whether to rerun with a different feature policy,

whether the branch should escalate from “AutoML baseline” to “custom model code.”

That is the key difference:

AutoML is good at optimizing inside a predefined search space.

The agent is good at redefining the search space, budget, constraints, and next hypothesis.

So I would not say “AutoML makes the agent unnecessary.” I would say:
AutoML shrinks the fragile part of the problem, so the agent can focus on the more semantic part.

3. Then what exactly can the LLM still tune after AutoGluon returns a good metric?

This is the right question.

If AutoGluon returns a strong result, the next search step should usually not be “randomly tune more.” It should be “change a different level of the problem.”

A good agent can still reason about:

changing the metric itself, since AutoGluon optimizes toward the eval metric you choose; the docs say it tunes factors like hyperparameters, early stopping, and ensemble weights to improve that metric on validation data.

changing the validation policy, because AutoGluon warns that tuning data is used to choose the best model and ensemble weights and should not be treated as fully unseen test data.

changing the allowed model set, because the hyperparameters argument is literally a dictionary of model keys and model-specific settings.

changing the search budget, because time_limit, presets, HPO trials, and search spaces materially affect what AutoGluon can do. The HPO tutorial also explicitly says you can call fit() multiple times while changing settings like time_limit, num_epochs, and num_boost_round to see how those choices affect outcomes.

changing feature and preprocessing policy, which is where your agent can add real value even if the predictor API is structured.

So the outer loop is still very meaningful even after AutoML runs.

4. Where does Optuna fit, then?

Optuna is a general optimization framework. Its core abstraction is simple: a study is an optimization task, and a trial is one execution of the objective function. It uses a define-by-run API, supports conditional search spaces, and supports pruning unpromising trials early via report() and should_prune().

That means Optuna fits very naturally in agentic pipelines, because the agent can decide:

what objective to optimize,

what parameters are exposed,

which trials are worth launching,

and when to stop the study.

Optuna is also framework-agnostic in the sense that it optimizes an objective function you write. Its official integrations include libraries like sklearn, PyTorch, TensorFlow, LightGBM, XGBoost, and MLflow. I did not find AutoGluon in Optuna’s official integration list, which suggests that AutoGluon + Optuna is possible mainly as a wrapper pattern, not as a first-class built-in integration.

So the practical answer is:

Yes, Optuna is compatible with an agentic pipeline.
Yes, it can also be compatible with AutoGluon.
But for AutoGluon, that compatibility is usually:
“wrap an AutoGluon run inside an Optuna objective,”
not “use a native official Optuna-AutoGluon plugin.”

That last part is my inference from the official integration docs.

5. So should the agent control AutoGluon, or should the agent write code itself?

For your kind of project, I would strongly recommend this rule:

Use AutoGluon for inner-loop execution by default.
Use LLM reasoning for outer-loop experiment design.
Use Optuna only when you want more explicit optimization control than AutoGluon’s built-in search gives you.
Allow free code editing only in a narrow, controlled zone.

Why I say that:

AI Scientist-v2 itself warns that it executes LLM-written code and should be run in a controlled sandbox, and its README explicitly says v2 is broader and more exploratory but has lower success rates than v1 when a strong starting template exists. v1 works best when there is a clear template and solid task foundation.

That maps very directly to your situation. You are not trying to build a frontier autonomous scientist. You are trying to build a working group project that also teaches you the architecture.

So a good hybrid is:

Mode A: AutoML-centered agent

The agent chooses:

task framing,

eval metric,

split policy,

candidate model families,

time budget,

feature policy,

whether to rerun.

AutoGluon handles the actual model fitting and ensembling. This is the safest mode.

Mode B: Optuna-centered agent

The agent chooses:

objective,

search space,

pruning policy,

constraints,

stopping rule.

Optuna runs the study. This is better when you already know the exact model family and want deeper control over the tuning logic.

Mode C: Limited code-editing agent

The agent may edit:

preprocessing modules,

feature generation code,

validation code,

thresholding/calibration logic,

task adapters.

But it should not freely rewrite the whole training engine at first.

That is the sweet spot.

6. What should the agent do if AutoML already found a strong model?

This is where the search layer matters.

Once AutoGluon has given you a strong incumbent, the next step should usually be one of these:

accept and deepen: spend more budget on the same family if it is still improving,

accept and constrain: keep score similar but reduce latency / complexity,

reject and redirect: if performance is unstable, validation looks suspicious, or guardrails are violated,

debug: only if the branch failed technically or produced invalid artifacts,

reopen exploration: if multiple branches are close or the incumbent plateaus.

That is the kind of acceptance/rejection logic the outer search layer owns. AI Scientist-v2’s BFTS config gives you the idea of breadth, steps, and debug depth; Optuna gives you the idea of pruning bad trials; AutoGluon gives you the inner evaluation results. Your agent stitches those together.

My own recommended starter rule would be:

warm-up ends after each candidate gets 2–3 valid runs,

optimize only the best valid branch,

debug only execution failures,

reopen search if the best branch stops improving beyond a small threshold.

That rule is my recommendation, not a claim from one specific repo.

7. The simplest way to combine all three without weirdness

Here is the cleanest non-weird architecture:

LLM agent
decides the next experiment plan

→ Search layer
tracks branches, budgets, accept/reject, warm-up vs optimization

→ Execution tool
either AutoGluon or Optuna-backed custom training

→ Run logs / case bank
store metrics, configs, rationale, failures, and next hypotheses

That works because each layer owns a different problem:

AutoGluon = powerful default executor,

Optuna = explicit optimizer when needed,

LLM agent = semantic planner and experiment manager.

My blunt final recommendation

For your first serious version, do not choose between “pure AutoML” and “pure LLM-written code.”

Build this instead:

Phase 1: AutoGluon baseline branch
Phase 2: agent modifies metric / split / model family / time budget / feature policy
Phase 3: Optuna branch only for one or two selected custom model families
Phase 4: limited code editing only in preprocessing and evaluation modules
Phase 5: free-form model-code editing only if you later want a research-style extension

That way:

you still get a real agentic loop,

you still learn experiment memory and search policy,

but you do not hand the whole project over to fragile code generation.


## My friend told me how to read the reference_repos 

My strong recommendation is:

Clone them into a separate reference_repos/ folder, not inside your main app package.

Read them as design references, not as something to directly merge.

Extract only:

orchestration patterns

experiment loop structure

config / logging style

search / optimization interfaces

evaluation workflow

result tracking ideas

What each repo is useful for
1) SakanaAI / AI-Scientist

This repo is useful if you want to study a full autonomous research loop: idea generation, experiment generation, code execution, evaluation, and paper/report production. The README explicitly describes it as a system for fully automatic scientific discovery, and its run flow centers around launch_scientist.py plus experiment templates.

What you should look for there:

Top-level workflow orchestration

How tasks are broken into stages

How prompts / roles / outputs are structured between stages

How experiment results are persisted and passed forward

How they separate “idea generation” from “execution”

What is worth borrowing:

a state-machine style pipeline

a directory convention for experiments, outputs, logs, and artifacts

an agent loop skeleton like:

propose idea

instantiate experiment

run code

evaluate result

reflect

revise next trial

For your project, this is more useful as an example of agent orchestration architecture than as ML training code.

2) SakanaAI / AI-Scientist-v2

This one is probably even more relevant to your agentic ML idea because it is explicitly framed as a generalized end-to-end agentic system and is associated with agentic tree search for scientific discovery.

What to study here:

search over candidate ideas / branches

branch scoring / selection

reflection and pruning logic

how they move from single-shot generation to structured search

This is where you can get inspiration for your earlier question about whether an agent should do something like:

continue search

stop and accept

backtrack

debug

refine a branch

allocate more budget to promising candidates

That is a very agentic layer.
It is not the same thing as AutoML.

Useful template ideas:

branch node structure

evaluation score objects

search budget control

retry / failure handling

“best-so-far” memory

tree expansion criteria

For your project, AI-Scientist-v2 is the best repo to study for the reasoning/search controller, while AutoGluon and Optuna are the best to study for the model optimization engine.

3) AutoGluon

AutoGluon is useful for the ML execution layer. Its official materials describe it as an AutoML system that can train strong tabular, text, image, and time-series models with only a few lines of code, and the project centers on automated model selection, ensembling, and training workflows.

What to look for:

predictor API design

fit / predict / leaderboard / evaluation flow

feature preprocessing pipeline

stacking / bagging / ensembling organization

trainer / learner separation

how configs are mapped to training behavior

how results and model metadata are stored

What is useful as a reference:

a clean abstraction like:

prepare_data()

fit_models()

evaluate_candidates()

select_best()

save_artifacts()

a model registry idea

standardized training result objects

good leaderboard / metrics reporting

how to expose many model families under one interface

For your agentic ML project, AutoGluon is the thing your agent may call as a tool.
The agent should not replace AutoGluon’s optimization internals; it should decide things like:

which dataset slice to use

which target / metric to optimize

whether to use AutoGluon tabular vs time-series

whether the result is good enough

whether to run another branch with different features / constraints / validation design

4) Optuna

Optuna is useful for the search engine for tunable decisions. The official docs describe it as a define-by-run hyperparameter optimization framework, and that flexibility is exactly why it is valuable inside agentic ML systems.

What to study:

objective(trial) pattern

dynamic search space definition

samplers and pruners

storage / trial history

intermediate metric reporting

early stopping logic

What is worth borrowing:

the study / trial abstraction

the separation of search logic from training logic

pruner-style thinking:

stop bad trials early

continue promising ones

explicit trial metadata logging

For your project, Optuna is ideal when your agent wants to optimize a bounded, explicit search space such as:

feature subset choices

model family choice

validation scheme choice

threshold tuning

prompt / tool policy parameters

retriever / reranker settings in an ML+agent system

So can you find useful code templates?

Yes — definitely. But the useful parts are mostly these:

Very useful to copy as ideas

project folder structure

experiment runner pattern

config-driven pipelines

result logging schema

trial / run metadata objects

evaluation summary objects

retry and failure handling

stage-by-stage orchestration

search tree / branch bookkeeping

leaderboard and artifact saving

Usually not worth directly copying

hard-coded prompts tied to their exact workflow

heavyweight experiment scripts tied to their benchmarks

repo-specific infra assumptions

full internal trainer stacks unless your project is very similar

code that exists mainly to support their paper/demo setup

The best way to use these repos in your project

A good architecture for you would be:

Layer 1: Agent controller

inspired by AI-Scientist / AI-Scientist-v2

decides what to try next

keeps memory of prior trials

compares branches

allocates budget

Layer 2: Optimization tools

AutoGluon for automated baseline modeling / stacking / ensembling

Optuna for explicit search spaces and trial pruning

Layer 3: Evaluation / reflection

unified metrics object

error analysis

overfitting checks

cost / latency / interpretability tradeoffs

Layer 4: Experiment memory

store:

config

data version

validation scheme

metric history

failure reasons

reflection notes

That combination is much better than “just import everything and hope it works.”

My honest view on each repo for your use case

If your goal is agentic ML for practical model development, then:

Most useful for orchestration ideas: AI-Scientist-v2

Most useful for practical AutoML execution: AutoGluon

Most useful for tunable search / HPO integration: Optuna

Most useful for full autonomous workflow inspiration: AI-Scientist v1

The main risk

The biggest mistake would be to treat these repos as if they are ready-made modules to paste together.

They are built with different assumptions:

AI-Scientist repos are about autonomous research workflows

AutoGluon is about AutoML execution

Optuna is about search / optimization primitives

So the right question is not “can I clone them and reuse code?”
The right question is:

which abstractions from each repo should become part of my own architecture?

That answer is yes.

A practical extraction checklist for your review:

find the main entrypoint

trace config loading

trace run-state objects

trace how one trial/experiment is represented

trace how evaluation is computed

trace how the next step is chosen

trace what gets logged and saved

That will give you much more value than reading random files.