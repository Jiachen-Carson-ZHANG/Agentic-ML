"""
phase3_execution.py
===================
Phase III — Execution Agents

Agents : CodeConfigAgent, TrainingExecutionAgent
Inputs : logs/trial_spec.json, logs/feature_table.parquet,
         trials/{trial_id}/trial_meta.json, train_df (DataFrame)
Outputs: trials/{trial_id}/uplift_scores.csv
         trials/{trial_id}/model.pkl  (or model_t.pkl + model_c.pkl for TwoModels)
         trials/{trial_id}/trial_meta.json  (updated with metrics / status)
"""

from __future__ import annotations

import json
import pickle
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from sklift.models import SoloModel, TwoModels, ClassTransformation
from sklift.metrics import uplift_auc_score, qini_auc_score, uplift_at_k

from experiment_planning import LEARNER_REGISTRY, ESTIMATOR_DEFAULTS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_LEARNER_FAMILIES = {"SoloModel", "TwoModels", "ResponseModel", "ClassTransformation"}
VALID_BASE_ESTIMATORS  = {"XGBoost", "LightGBM", "CatBoost"}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    trial_id: str
    status: str                      # running | complete | failed
    learner_family: str
    base_estimator: str
    feature_recipe: str
    params: dict[str, Any]
    split_seed: int
    metrics: dict[str, float]        # auuc, qini, uplift_at_10, uplift_at_20
    training_time_seconds: float
    uplift_scores_path: str
    model_paths: list[str]
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal model factories
# ---------------------------------------------------------------------------

def _build_base_estimator(base_estimator: str, params: dict):
    """Instantiate the correct sklearn-compatible classifier."""
    p = {k: v for k, v in params.items()}

    if base_estimator == "XGBoost":
        from xgboost import XGBClassifier
        p.pop("use_label_encoder", None)  # deprecated in newer xgboost
        return XGBClassifier(**p)

    if base_estimator == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**p)

    if base_estimator == "CatBoost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**p)

    raise ValueError(f"Unsupported base_estimator: {base_estimator}")


def _build_uplift_model(learner_family: str, base_estimator: str, params: dict):
    """Wrap the base estimator in the correct sklift uplift model class."""
    clf = _build_base_estimator(base_estimator, params)

    if learner_family in ("SoloModel", "ResponseModel"):
        # sklift has no standalone ResponseModel; treatment_interaction is the
        # closest equivalent (adds treatment flag + interaction terms as features)
        return SoloModel(estimator=clf, method="treatment_interaction")

    if learner_family == "TwoModels":
        clf_t = _build_base_estimator(base_estimator, params)
        clf_c = _build_base_estimator(base_estimator, params)
        return TwoModels(estimator_trmnt=clf_t, estimator_ctrl=clf_c, method="vanilla")

    if learner_family == "ClassTransformation":
        return ClassTransformation(estimator=clf)

    raise ValueError(f"Unsupported learner_family: {learner_family}")


# ---------------------------------------------------------------------------
# Agent 1: CodeConfigAgent
# ---------------------------------------------------------------------------

class CodeConfigAgent:
    """
    Reads logs/trial_spec.json, validates all required fields, and returns
    a resolved config dict ready for TrainingExecutionAgent.

    Raises ValueError with a clear message on any missing or invalid field.
    """

    def __init__(self, trial_spec_path: str = "logs/trial_spec.json"):
        self.trial_spec_path = Path(trial_spec_path)

    def run(self) -> dict[str, Any]:
        if not self.trial_spec_path.exists():
            raise FileNotFoundError(
                f"trial_spec.json not found at {self.trial_spec_path}"
            )

        with self.trial_spec_path.open() as f:
            spec = json.load(f)

        # --- Validate learner_family ---
        learner_family = spec.get("learner_family")
        if not learner_family:
            model_str = spec.get("model", "")
            for fam in VALID_LEARNER_FAMILIES:
                if fam in model_str:
                    learner_family = fam
                    break
        if learner_family not in VALID_LEARNER_FAMILIES:
            raise ValueError(
                f"Invalid or missing learner_family '{learner_family}'. "
                f"Must be one of: {sorted(VALID_LEARNER_FAMILIES)}"
            )

        # --- Validate base_estimator ---
        base_estimator = spec.get("base_estimator")
        if not base_estimator:
            model_str = spec.get("model", "")
            for est in VALID_BASE_ESTIMATORS:
                if est in model_str:
                    base_estimator = est
                    break
        if base_estimator not in VALID_BASE_ESTIMATORS:
            raise ValueError(
                f"Invalid or missing base_estimator '{base_estimator}'. "
                f"Must be one of: {sorted(VALID_BASE_ESTIMATORS)}"
            )

        # --- Required fields ---
        missing = [k for k in ("trial_id", "feature_recipe", "params") if not spec.get(k)]
        if missing:
            raise ValueError(
                f"trial_spec.json is missing required fields: {missing}"
            )

        return {
            "trial_id":       spec["trial_id"],
            "learner_family": learner_family,
            "base_estimator": base_estimator,
            "feature_recipe": spec["feature_recipe"],
            "params":         spec["params"],
            "split_seed":     spec.get("split_seed", 42),
            "hypothesis":     spec.get("hypothesis", ""),
            "hypothesis_id":  spec.get("hypothesis_id", ""),
        }


# ---------------------------------------------------------------------------
# Agent 2: TrainingExecutionAgent
# ---------------------------------------------------------------------------

class TrainingExecutionAgent:
    """
    Merges the feature table with train_df, splits stratified by treatment_flg,
    trains the uplift model, computes metrics, and writes all output artefacts
    to trials/{trial_id}/.

    On any exception: writes status=failed to trial_meta.json and returns a
    TrainingResult with status='failed' instead of raising.
    """

    def __init__(
        self,
        feature_table_path: str,
        trial_meta_path: str,
        train_df: pd.DataFrame,
        config: dict[str, Any],
    ):
        self.feature_table_path = Path(feature_table_path)
        self.trial_meta_path    = Path(trial_meta_path)
        self.train_df           = train_df.copy()
        self.config             = config

    def run(self) -> TrainingResult:
        trial_id       = self.config["trial_id"]
        learner_family = self.config["learner_family"]
        base_estimator = self.config["base_estimator"]
        params         = self.config.get("params", {})
        split_seed     = self.config.get("split_seed", 42)
        feature_recipe = self.config.get("feature_recipe", "")

        output_dir = self.trial_meta_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write status=running immediately — before any training begins
        self._write_meta_status(trial_id, "running")

        try:
            # ---- Load and merge features ----
            feature_df = pd.read_parquet(self.feature_table_path)
            if "client_id" in feature_df.columns and "customer_id" not in feature_df.columns:
                feature_df = feature_df.rename(columns={"client_id": "customer_id"})

            id_col = "client_id" if "client_id" in self.train_df.columns else "customer_id"
            merged = (
                self.train_df
                .rename(columns={id_col: "customer_id"})
                .merge(feature_df, on="customer_id", how="inner")
            )

            # ---- Feature matrix — never include target or treatment_flg ----
            exclude      = {"customer_id", "target", "treatment_flg"}
            feature_cols = [c for c in merged.columns if c not in exclude]
            X = merged[feature_cols].values.astype(float)
            y = merged["target"].values.astype(int)
            t = merged["treatment_flg"].values.astype(int)

            # ---- Stratified train/test split (stratify by treatment_flg) ----
            X_tr, X_te, y_tr, y_te, t_tr, t_te, idx_tr, idx_te = train_test_split(
                X, y, t, merged.index,
                test_size=0.2,
                random_state=split_seed,
                stratify=t,
            )
            test_ids = merged.loc[idx_te, "customer_id"].values

            # ---- Train ----
            model   = _build_uplift_model(learner_family, base_estimator, params)
            t_start = time.time()
            model.fit(X_tr, y_tr, t_tr)
            training_time = time.time() - t_start

            # ---- Score + metrics ----
            scores = model.predict(X_te)
            metrics = {
                "auuc":        float(uplift_auc_score(y_te, scores, t_te)),
                "qini":        float(qini_auc_score(y_te, scores, t_te)),
                "uplift_at_10": float(uplift_at_k(y_te, scores, t_te, strategy="overall", k=0.1)),
                "uplift_at_20": float(uplift_at_k(y_te, scores, t_te, strategy="overall", k=0.2)),
            }

            # ---- Save uplift_scores.csv ----
            scores_path = output_dir / "uplift_scores.csv"
            pd.DataFrame({
                "client_id":    test_ids,
                "uplift_score": scores,
                "treatment":    t_te,
                "target":       y_te,
            }).to_csv(scores_path, index=False)

            # ---- Save model pickle(s) ----
            model_paths: list[str] = []
            if learner_family == "TwoModels":
                p_t = output_dir / "model_t.pkl"
                p_c = output_dir / "model_c.pkl"
                with p_t.open("wb") as fh:
                    pickle.dump(model.estimator_trmnt, fh)
                with p_c.open("wb") as fh:
                    pickle.dump(model.estimator_ctrl, fh)
                model_paths = [str(p_t), str(p_c)]
            else:
                p_m = output_dir / "model.pkl"
                with p_m.open("wb") as fh:
                    pickle.dump(model, fh)
                model_paths = [str(p_m)]

            # ---- Update trial_meta.json ----
            self._write_meta_complete(trial_id, metrics, training_time)

            return TrainingResult(
                trial_id=trial_id,
                status="complete",
                learner_family=learner_family,
                base_estimator=base_estimator,
                feature_recipe=feature_recipe,
                params=params,
                split_seed=split_seed,
                metrics=metrics,
                training_time_seconds=training_time,
                uplift_scores_path=str(scores_path),
                model_paths=model_paths,
            )

        except Exception as exc:
            self._write_meta_failed(trial_id, str(exc))
            return TrainingResult(
                trial_id=trial_id,
                status="failed",
                learner_family=learner_family,
                base_estimator=base_estimator,
                feature_recipe=feature_recipe,
                params=params,
                split_seed=split_seed,
                metrics={},
                training_time_seconds=0.0,
                uplift_scores_path="",
                model_paths=[],
                error_message=str(exc),
            )

    # ------------------------------------------------------------------
    # trial_meta.json helpers
    # ------------------------------------------------------------------

    def _read_meta(self) -> dict:
        if self.trial_meta_path.exists():
            with self.trial_meta_path.open() as f:
                return json.load(f)
        return {}

    def _write_meta(self, data: dict) -> None:
        self.trial_meta_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trial_meta_path.open("w") as f:
            json.dump(data, f, indent=2)

    def _write_meta_status(self, trial_id: str, status: str) -> None:
        meta = self._read_meta()
        meta.update({"trial_id": trial_id, "status": status,
                     "updated_at": datetime.utcnow().isoformat()})
        self._write_meta(meta)

    def _write_meta_complete(
        self, trial_id: str, metrics: dict, training_time: float
    ) -> None:
        meta = self._read_meta()
        meta.update({
            "trial_id": trial_id,
            "status": "complete",
            "metrics": metrics,
            "training_time_seconds": training_time,
            "completed_at": datetime.utcnow().isoformat(),
        })
        self._write_meta(meta)

    def _write_meta_failed(self, trial_id: str, error_message: str) -> None:
        meta = self._read_meta()
        meta.update({
            "trial_id": trial_id,
            "status": "failed",
            "error_message": error_message,
            "failed_at": datetime.utcnow().isoformat(),
        })
        self._write_meta(meta)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    import uuid
    import numpy as np

    print("=" * 60)
    print("Phase III smoke test — CodeConfigAgent + TrainingExecutionAgent")
    print("=" * 60)

    RNG = np.random.default_rng(0)
    N   = 20

    stub_train = pd.DataFrame({
        "client_id":     [f"C{i:03d}" for i in range(N)],
        "target":        RNG.integers(0, 2, N).tolist(),
        "treatment_flg": [1] * (N // 2) + [0] * (N // 2),
    })
    stub_features = pd.DataFrame(RNG.random((N, 10)), columns=[f"f{i}" for i in range(10)])
    stub_features.insert(0, "customer_id", [f"C{i:03d}" for i in range(N)])

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        logs_dir = tmp / "logs"
        logs_dir.mkdir()

        feat_path = logs_dir / "feature_table.parquet"
        stub_features.to_parquet(feat_path, index=False)

        trial_id  = str(uuid.uuid4())
        spec_dict = {
            "trial_id":       trial_id,
            "learner_family": "SoloModel",
            "base_estimator": "XGBoost",
            "feature_recipe": "rfm_baseline",
            "params":         {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1},
            "split_seed":     42,
            "hypothesis":     "smoke test",
        }
        spec_path = logs_dir / "trial_spec.json"
        spec_path.write_text(json.dumps(spec_dict, indent=2))

        trials_dir = tmp / "trials" / trial_id
        trials_dir.mkdir(parents=True)
        meta_path  = trials_dir / "trial_meta.json"
        meta_path.write_text(json.dumps({"trial_id": trial_id, "status": "pending"}))

        print("\n[1/2] CodeConfigAgent")
        config = CodeConfigAgent(trial_spec_path=str(spec_path)).run()
        assert config["learner_family"] == "SoloModel"
        assert config["base_estimator"] == "XGBoost"
        print(f"  learner_family={config['learner_family']}  base_estimator={config['base_estimator']}")

        print("\n[2/2] TrainingExecutionAgent")
        result = TrainingExecutionAgent(
            feature_table_path=str(feat_path),
            trial_meta_path=str(meta_path),
            train_df=stub_train,
            config=config,
        ).run()
        print(f"  status={result.status}")
        if result.status == "complete":
            print(f"  AUUC={result.metrics['auuc']:.4f}  Qini={result.metrics['qini']:.4f}")
        else:
            print(f"  error={result.error_message}")

        # Verify output files exist
        assert (trials_dir / "uplift_scores.csv").exists()
        assert (trials_dir / "model.pkl").exists()
        meta_written = json.loads((trials_dir / "trial_meta.json").read_text())
        assert meta_written["status"] in ("complete", "failed")

        print("\n" + "=" * 60)
        print("Phase III smoke test PASSED")
        print("=" * 60)
