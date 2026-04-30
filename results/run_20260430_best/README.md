# Best Autonomous Run — 2026-04-30

**Champion:** class_transformation + LightGBM on `hybrid_safe_semantic_v1`
- Held-out Normalized Qini: **0.337** (vs human notebook best 0.328)
- Val/HO gap: −1.7 raw Qini (near-zero — highly stable)
- XAI top features: age_clean, days_to_first_redeem, points_received_total_30d

## Files

| File | What it contains |
|---|---|
| `pipeline.log` | Full stage-by-stage run trace including `[plan]` (o3 hypothesis) and `[eval]` (o4-mini verdict) lines for every trial |
| `uplift_ledger.jsonl` | Append-only trial records: metrics, verdict, judge_narrative, xai_summary, policy_narrative, strategy_rationale |
| `final_report.md` | Auto-generated report: champion, benchmark, all trials table, policy recommendation, XAI explanation |
| `hypotheses.jsonl` | Hypothesis lifecycle records |

## Trial Summary

| Family | Estimator | Recipe | HO Norm Qini | Gap | Verdict |
|---|---|---|---:|---:|---|
| two_model | logistic_regression | rfm_baseline | 0.248 | +0.054 | baseline |
| class_transformation | gradient_boosting | rfm_baseline | 0.187 | −0.074 | inconclusive |
| two_model | xgboost | hybrid_safe_semantic_v1 | 0.249 | −0.124 | supported |
| class_transformation | xgboost | hybrid_safe_semantic_v1 | 0.313 | −0.049 | supported |
| two_model | lightgbm | hybrid_safe_semantic_v1 | 0.096 | −0.156 | inconclusive |
| **class_transformation** | **lightgbm** | **hybrid_safe_semantic_v1** | **0.337** | **−0.005** | **supported** |

## Key Finding

Switching from `rfm_baseline` (27k pre-issue transactions) to `hybrid_safe_semantic_v1`
(3.9M post-issue transactions with audited cutoff) shifted XAI top features from
pure demographics (`age_clean` only) to include behavioral signals
(`days_to_first_redeem`, `points_received_total_30d`), and improved holdout
Qini by +22 raw points over the baseline.

The agent's verdict ceiling correctly blocked two overfitting trials (inconclusive)
and selected the most stable champion via stability-adjusted scoring.
