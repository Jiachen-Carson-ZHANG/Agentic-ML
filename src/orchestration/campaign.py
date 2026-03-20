"""
Outer optimization loop over multiple ExperimentSessions.

Origin  : campaign.py entrypoint (repo root)
Consumed: nothing downstream — writes campaign.json and campaign.log to disk

Each iteration:
  1. PreprocessingExecutor applies PreprocessingPlan (Phase 4a: identity)
  2. ExperimentSession runs warm-up + optimize loop on the preprocessed data
  3. CampaignOrchestrator records SessionSummary, checks stop conditions

Sessions are stored inside campaigns/{campaign_id}/sessions/ so each campaign
is a self-contained folder.
"""
from __future__ import annotations
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

from src.models.campaign import CampaignConfig, SessionSummary, CampaignResult
from src.models.task import TaskSpec
from src.models.preprocessing import PreprocessingPlan
from src.llm.backend import LLMBackend
from src.execution.preprocessing_runner import PreprocessingExecutor
from src.session import ExperimentSession


class CampaignOrchestrator:
    """
    Runs multiple ExperimentSessions on the same task, stopping when the
    metric plateaus or the session budget is exhausted.

    Sessions are stored in campaigns/{campaign_id}/sessions/ for easy navigation.
    campaign.json is written after each session so partial results survive crashes.

    Phase 4a: always uses identity preprocessing.
    Phase 4b: will generate new preprocessing strategies on plateau.
    """

    def __init__(
        self,
        task: TaskSpec,
        llm: LLMBackend,
        config: Optional[CampaignConfig] = None,
        experiments_dir: str = "experiments",
        num_candidates: int = 3,
        max_optimize_iterations: int = 5,
        higher_is_better: bool = True,
        case_store_path: Optional[str] = None,
    ) -> None:
        self._task = task
        self._llm = llm
        self._config = config or CampaignConfig()
        self._experiments_dir = experiments_dir
        self._num_candidates = num_candidates
        self._max_optimize_iterations = max_optimize_iterations
        self._higher_is_better = higher_is_better
        self._case_store_path = case_store_path
        self._executor = PreprocessingExecutor()

    def run(self) -> CampaignResult:
        campaign_id = str(uuid.uuid4())[:8]
        started_at = datetime.now().isoformat()
        campaign_dir = Path(self._experiments_dir) / "campaigns" / f"{campaign_id}_{self._task.task_name}"
        sessions_dir = campaign_dir / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        # Campaign-level logger: writes to both stdout and campaign.log
        log = logging.getLogger(f"campaign.{campaign_id}")
        log.setLevel(logging.DEBUG)
        if not log.handlers:
            fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            fh = logging.FileHandler(campaign_dir / "campaign.log", mode="a")
            fh.setFormatter(fmt)
            log.addHandler(sh)
            log.addHandler(fh)

        log.info("=" * 60)
        log.info(f"Campaign {campaign_id} | task={self._task.task_name} | max_sessions={self._config.max_sessions}")
        log.info("=" * 60)

        sessions: List[SessionSummary] = []
        metrics: List[float] = []

        for i in range(self._config.max_sessions):
            log.info(f"--- Session {i + 1}/{self._config.max_sessions} ---")
            t_start = datetime.now()
            summary: Optional[SessionSummary] = None

            try:
                plan = self._preprocessing_plan()
                prep_dir = campaign_dir / f"preprocessing_{i + 1}"
                preprocessed_path = self._executor.run(self._task.data_path, plan, str(prep_dir))

                session = ExperimentSession(
                    task=self._task,
                    llm=self._llm,
                    experiments_dir=str(sessions_dir),
                    num_candidates=self._num_candidates,
                    max_optimize_iterations=self._max_optimize_iterations,
                    higher_is_better=self._higher_is_better,
                    case_store_path=self._case_store_path,
                    preprocessed_data_path=preprocessed_path,
                )
                session.run()

                duration = (datetime.now() - t_start).total_seconds()
                incumbent = session.run_store.get_incumbent(higher_is_better=self._higher_is_better)
                best_metric = incumbent.result.primary_metric if incumbent else None

                summary = SessionSummary(
                    session_id=str(session._session_dir.name),
                    best_metric=best_metric,
                    preprocessing_strategy=plan.strategy,
                    session_dir=str(session._session_dir),
                    duration_seconds=duration,
                )
                if best_metric is not None:
                    metrics.append(best_metric)
                    log.info(f"Session {i + 1} best: {best_metric:.4f}")
                else:
                    log.warning(f"Session {i + 1}: no successful runs")

            except Exception as exc:
                duration = (datetime.now() - t_start).total_seconds()
                log.error(f"Session {i + 1} failed: {exc}")
                summary = SessionSummary(
                    session_id=f"session_{i + 1}_failed",
                    best_metric=None,
                    preprocessing_strategy="identity",
                    session_dir="",
                    duration_seconds=duration,
                    error_message=str(exc),
                )

            sessions.append(summary)
            # Write campaign.json after every session so partial results survive
            partial = self._build_result(campaign_id, started_at, sessions, "budget")
            self._save(partial, campaign_dir)

            if self._is_plateau(metrics):
                log.info("Plateau detected — stopping campaign.")
                result = self._build_result(campaign_id, started_at, sessions, "plateau")
                self._save(result, campaign_dir)
                return result

        result = self._build_result(campaign_id, started_at, sessions, "budget")
        self._save(result, campaign_dir)
        log.info(f"Campaign complete: best={result.best_metric} | stopped={result.stopped_reason}")
        return result

    def _is_plateau(self, metrics: List[float]) -> bool:
        """True if the last plateau_window metrics are all within plateau_threshold of each other."""
        if len(metrics) < self._config.plateau_window:
            return False
        recent = metrics[-self._config.plateau_window:]
        return max(recent) - min(recent) < self._config.plateau_threshold

    def _best_metric(self, metrics: List[float]) -> Optional[float]:
        """Returns best metric respecting higher_is_better."""
        if not metrics:
            return None
        return max(metrics) if self._higher_is_better else min(metrics)

    def _preprocessing_plan(self) -> PreprocessingPlan:
        """Phase 4a: always identity. Phase 4b: call PreprocessingAgent here."""
        return PreprocessingPlan(strategy="identity")

    def _build_result(
        self,
        campaign_id: str,
        started_at: str,
        sessions: List[SessionSummary],
        stopped_reason: Literal["plateau", "budget"],
    ) -> CampaignResult:
        metrics_with_values = [s.best_metric for s in sessions if s.best_metric is not None]
        best_metric = self._best_metric(metrics_with_values)
        best_session_id = None
        if best_metric is not None:
            best_session_id = next(
                s.session_id for s in sessions if s.best_metric == best_metric
            )
        return CampaignResult(
            campaign_id=campaign_id,
            task_name=self._task.task_name,
            started_at=started_at,
            sessions=sessions,
            best_metric=best_metric,
            best_session_id=best_session_id,
            stopped_reason=stopped_reason,
        )

    def _save(self, result: CampaignResult, campaign_dir: Path) -> None:
        path = campaign_dir / "campaign.json"
        path.write_text(result.model_dump_json(indent=2))
