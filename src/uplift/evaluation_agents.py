"""PR2-style post-training evaluation phase."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from src.models.uplift import UpliftExperimentRecord
from src.uplift.ledger import UpliftLedger
from src.uplift.llm_client import ChatLLM
from src.uplift.metrics import qini_auc_score, uplift_at_k, uplift_auc_score
from src.uplift.policy import build_policy_summary
from src.uplift.xai import (
    check_leakage_signals,
    explain_cached_uplift_model,
    explain_score_feature_associations,
    run_shap_solo_model,
    run_shap_two_model,
)

_SKILLS_DIR = Path(__file__).parent / "skills"


def _load_skill(name: str) -> str:
    path = _SKILLS_DIR / f"{name}.md"
    if not path.exists():
        return name.replace("_", " ")
    text = path.read_text(encoding="utf-8")
    match = re.search(r"## System Prompt\s*\n+```[^\n]*\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _parse_json(text: str) -> dict:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return {}
    return {}


def _scores_to_arrays(scores_df: pd.DataFrame):
    return (
        scores_df["target"].values,
        scores_df["treatment_flg"].values,
        scores_df["uplift"].values,
    )


class UpliftEvaluationJudge:
    """Decide whether the tested hypothesis is supported, refuted, or inconclusive."""

    _SKILL = _load_skill("evaluation_judge")

    def __init__(self, llm: ChatLLM) -> None:
        self.llm = llm

    def run(
        self,
        trial_meta: dict,
        scores_df: pd.DataFrame,
        prior_champion: UpliftExperimentRecord | None = None,
        stability_score: float = 1.0,
    ) -> dict:
        y_true, treatment, uplift = _scores_to_arrays(scores_df)
        metrics = {
            "qini_auc": qini_auc_score(y_true, treatment, uplift),
            "uplift_auc": uplift_auc_score(y_true, treatment, uplift),
            "uplift_at_5pct": uplift_at_k(y_true, treatment, uplift, k=0.05),
            "uplift_at_10pct": uplift_at_k(y_true, treatment, uplift, k=0.10),
            "uplift_at_20pct": uplift_at_k(y_true, treatment, uplift, k=0.20),
        }
        champion_block = (
            {
                "qini_auc": prior_champion.qini_auc,
                "uplift_auc": prior_champion.uplift_auc,
                "verdict": prior_champion.verdict,
            }
            if prior_champion is not None
            else "None - first run."
        )
        parsed = _parse_json(
            self.llm(
                self._SKILL,
                json.dumps(
                    {
                        "trial_meta": trial_meta,
                        "computed_metrics": metrics,
                        "stability_score": stability_score,
                        "prior_champion": champion_block,
                    },
                    sort_keys=True,
                ),
            )
        )
        parsed.setdefault("verdict", "inconclusive")
        parsed["computed_metrics"] = metrics
        parsed["trial_id"] = trial_meta.get("trial_id") or trial_meta.get("spec_id")
        return parsed


class UpliftXAIReasoner:
    """Run optional XAI and summarize leakage/plausibility evidence."""

    _SKILL = _load_skill("xai_reasoning")

    def __init__(self, llm: ChatLLM) -> None:
        self.llm = llm

    def run(
        self,
        trial_meta: dict,
        features_df: pd.DataFrame,
        model_dir: Path | None = None,
        judge_verdict: dict | None = None,
        scores_df: pd.DataFrame | None = None,
    ) -> dict:
        cached_model_result = self._try_cached_model_xai(
            features_df,
            model_dir,
            scores_df,
        )
        trial_id = trial_meta.get("trial_id") or trial_meta.get("spec_id")
        if cached_model_result is not None:
            cached_model_result["trial_id"] = trial_id
            cached_model_result["skipped"] = False
            cached_model_result["leakage_auto_flag"] = check_leakage_signals(
                {"top_features": cached_model_result["global_top_features"]}
            )
            return cached_model_result

        shap_result = self._try_shap(
            trial_meta.get("learner_family", "solo_model"),
            features_df,
            model_dir,
        )
        if shap_result is None:
            if scores_df is not None and not features_df.empty:
                fallback = explain_score_feature_associations(features_df, scores_df)
                fallback["trial_id"] = trial_id
                fallback["skipped"] = False
                fallback["leakage_auto_flag"] = check_leakage_signals(
                    {"top_features": fallback["global_top_features"]}
                )
                return fallback
            return {
                "skipped": True,
                "reason": "Model files not available; XAI skipped.",
                "trial_id": trial_id,
            }

        leakage = check_leakage_signals(shap_result)
        parsed = _parse_json(
            self.llm(
                self._SKILL,
                json.dumps(
                    {
                        "trial_meta": trial_meta,
                        "hypothesis_text": trial_meta.get("hypothesis_text", ""),
                        "shap_result": shap_result,
                        "leakage_auto_flag": leakage,
                        "judge_verdict": judge_verdict,
                    },
                    sort_keys=True,
                ),
            )
        )
        parsed["shap_raw"] = shap_result
        parsed["trial_id"] = trial_id
        parsed["skipped"] = False
        return parsed

    def _try_shap(
        self,
        model_type: str,
        features_df: pd.DataFrame,
        model_dir: Path | None,
    ) -> dict | None:
        if model_dir is None or features_df.empty:
            return None
        try:
            if model_type == "two_model":
                treatment_model = model_dir / "model_t.pkl"
                control_model = model_dir / "model_c.pkl"
                if not (treatment_model.exists() and control_model.exists()):
                    return None
                return run_shap_two_model(treatment_model, control_model, features_df)
            model_path = model_dir / "model.pkl"
            if not model_path.exists():
                return None
            return run_shap_solo_model(model_path, features_df)
        except Exception:
            return None

    def _try_cached_model_xai(
        self,
        features_df: pd.DataFrame,
        model_dir: Path | None,
        scores_df: pd.DataFrame | None,
    ) -> dict | None:
        if model_dir is None or features_df.empty:
            return None
        model_path = model_dir / "model.pkl"
        if not model_path.exists():
            return None
        try:
            return explain_cached_uplift_model(
                model_path,
                features_df,
                scores_df,
            )
        except Exception:
            return None


class UpliftPolicyAdvisor:
    """Convert uplift scores into targeting policy recommendations."""

    _SKILL = _load_skill("policy_simulation")

    def __init__(self, llm: ChatLLM) -> None:
        self.llm = llm

    def run(
        self,
        trial_meta: dict,
        scores_df: pd.DataFrame,
        xai_result: dict | None = None,
        coupon_cost: float = 1.0,
        revenue_per_conversion: float = 10.0,
        budget: float | None = None,
    ) -> dict:
        policy_data = build_policy_summary(
            scores_df,
            coupon_cost=coupon_cost,
            revenue_per_conversion=revenue_per_conversion,
            budget=budget,
        )
        parsed = _parse_json(
            self.llm(
                self._SKILL,
                json.dumps(
                    {
                        "trial_meta": trial_meta,
                        "targeting_results": policy_data["targeting_results"],
                        "budget_result": policy_data["budget_result"],
                        "segment_summary": policy_data["segment_summary"],
                        "elbow_threshold_pct": policy_data["elbow_threshold_pct"],
                        "xai_findings": xai_result,
                    },
                    sort_keys=True,
                ),
            )
        )
        parsed.setdefault("recommended_threshold", policy_data["elbow_threshold_pct"])
        parsed["policy_data"] = policy_data
        parsed["trial_id"] = trial_meta.get("trial_id") or trial_meta.get("spec_id")
        return parsed


def run_evaluation_phase(
    trial_meta: dict,
    scores_df: pd.DataFrame,
    ledger: UpliftLedger,
    llm: ChatLLM,
    model_dir: Path | None = None,
    features_df: pd.DataFrame | None = None,
    coupon_cost: float = 1.0,
    revenue_per_conversion: float = 10.0,
    budget: float | None = None,
) -> dict:
    """Run PR2 Judge, XAI, and Policy agents for one completed trial."""
    records = ledger.load()
    champion = max(
        (record for record in records if record.status == "success" and record.qini_auc is not None),
        key=lambda record: record.qini_auc,
        default=None,
    )
    judge = UpliftEvaluationJudge(llm)
    xai = UpliftXAIReasoner(llm)
    policy = UpliftPolicyAdvisor(llm)

    judge_result = judge.run(trial_meta, scores_df, champion)
    xai_result = xai.run(
        trial_meta,
        features_df if features_df is not None else pd.DataFrame(),
        model_dir,
        judge_result,
        scores_df,
    )
    policy_result = policy.run(
        trial_meta,
        scores_df,
        xai_result,
        coupon_cost,
        revenue_per_conversion,
        budget,
    )
    return {"judge": judge_result, "xai": xai_result, "policy": policy_result}
