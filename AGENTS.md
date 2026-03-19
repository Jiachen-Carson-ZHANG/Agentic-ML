# MediGenius Project Rules

## Project goal
Build a clear, maintainable codebase for a high-quality medical chatbot with:
- a strong RAG pipeline
- reliable evals
- strong observability
- clear architecture
- minimal hardcoding
- production-minded iteration instead of patchy local fixes

## Working principles
- Prefer global optimization over local patching.
- Do not add quick fixes that increase architectural confusion.
- Favor simpler flows over adding more branches, flags, or hidden coupling.
- Keep modules small, readable, and single-purpose where practical.
- Prefer explicit contracts over implicit shared state.

## Change policy
- Before modifying code, identify the owning module and the interfaces affected.
- Before fixing a bug, check whether the real issue is architectural rather than local.
- When changing workflow, cache, retrieval, eval, or persistence logic, explain the intended system-level impact.
- Remove dead code, temporary branches, and obsolete experiments once replaced.

## RAG rules
- Keep retrieval, reranking, answer generation, orchestration, and eval logic clearly separated.
- Do not mix experiment code directly into the production path without a clear boundary.
- Any retrieval change should state what retrieval behavior is expected to improve.

## Cache and memory rules
- Do not introduce a new cache unless the key strategy, scope, invalidation rule, and observability are clear.
- Cache behavior should be version-aware when prompts, embeddings, models, or retrieval config change.
- Distinguish clearly between:
  - retrieval storage / vector index
  - app state / persistence
  - semantic cache
  - conversational memory

## Eval rules
- Do not claim an improvement without evaluation evidence.
- Separate retrieval evaluation from end-to-end answer/workflow evaluation.
- When fixing an important failure mode, add or update a regression case.
- Prefer small, trusted eval sets over large, noisy eval sets.

## Observability rules
- Important workflow branches should emit useful logs, metrics, or traces.
- Avoid swallowing exceptions silently.
- Broad exception handling is allowed only if there is clear logging and fallback intent.

## Refactor rules
- Split oversized files when they carry multiple responsibilities.
- Reduce duplication before adding new complexity.
- Prefer typed, explicit interfaces over hidden coupling and hardcoded assumptions.
- Keep configuration centralized where possible.

## Documentation rules
- Update `docs/architecture/current-state.md` when the system design changes materially.
- Update `docs/changes/implementation-log.md` for meaningful changes.
- Add an ADR in `docs/decisions/` for major architectural decisions.
- Keep docs short, concrete, and current.

## Definition of done
A change is not done unless:
- the code path is understandable
- the architectural impact is acceptable
- the relevant docs are updated if needed
- the change is testable or evaluable
- obvious dead code or outdated comments introduced by the change are cleaned up

## Project source-of-truth docs
- Architecture: `docs/architecture/current-state.md`
- Decisions: `docs/decisions/`
- Implementation log: `docs/changes/implementation-log.md`

## Documentation usage rules
- Before changing architecture, workflow, cache, retrieval, eval, observability, or persistence logic, read the relevant docs if they exist.
- When a material architecture change is made, update `docs/architecture/current-state.md`.
- When a major technical decision is made, add or update an ADR in `docs/decisions/`.
- When a meaningful implementation change is completed, append a concise entry to `docs/changes/implementation-log.md`.
- Do not treat chat history as the source of truth for project design; the docs folder is the persistent source of truth.
