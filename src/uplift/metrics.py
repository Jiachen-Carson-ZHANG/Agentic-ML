"""Uplift metric primitives for campaign targeting evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence
import warnings

import numpy as np
import pandas as pd

from src.models.uplift import UpliftEvaluationPolicy


@dataclass(frozen=True)
class UpliftMetricResult:
    """Computed uplift metrics plus tabular artifacts."""

    qini_auc: float
    uplift_auc: float
    uplift_at_k: Dict[str, float]
    policy_gain: Dict[str, float]
    decile_table: pd.DataFrame
    qini_curve: pd.DataFrame
    uplift_curve: pd.DataFrame


def _as_1d_array(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return arr


def _validate_uplift_inputs(
    y_true: Sequence[int] | np.ndarray,
    treatment: Sequence[int] | np.ndarray,
    uplift: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = _as_1d_array(y_true, "y_true").astype(int)
    t = _as_1d_array(treatment, "treatment").astype(int)
    u = _as_1d_array(uplift, "uplift").astype(float)
    if not (len(y) == len(t) == len(u)):
        raise ValueError("y_true, treatment, and uplift must have the same length")
    if len(y) == 0:
        raise ValueError("uplift metrics require at least one row")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("y_true must be binary 0/1")
    if not set(np.unique(t)).issubset({0, 1}):
        raise ValueError("treatment must be binary 0/1")
    if len(np.unique(t)) < 2:
        raise ValueError("uplift metrics require both treatment and control rows")
    if np.isnan(u).any():
        raise ValueError("uplift contains NaN values")
    return y, t, u


def _trapz_compatible(y_values: pd.Series, x_values: pd.Series) -> float:
    """Integrate with numpy 1.26 compatibility and quiet numpy 2.x deprecation."""
    # Keep np.trapz for numpy 1.26 compatibility; np.trapezoid is unavailable there.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="`trapz` is deprecated.*",
            category=DeprecationWarning,
        )
        return float(np.trapz(y_values, x_values))


def _sorted_frame(
    y_true: Sequence[int] | np.ndarray,
    treatment: Sequence[int] | np.ndarray,
    uplift: Sequence[float] | np.ndarray,
) -> pd.DataFrame:
    y, t, u = _validate_uplift_inputs(y_true, treatment, uplift)
    return pd.DataFrame({"target": y, "treatment": t, "uplift": u}).sort_values(
        "uplift",
        ascending=False,
        kind="mergesort",
    )


def qini_curve_data(
    y_true: Sequence[int] | np.ndarray,
    treatment: Sequence[int] | np.ndarray,
    uplift: Sequence[float] | np.ndarray,
) -> pd.DataFrame:
    """Return cumulative Qini curve points sorted by predicted uplift."""
    frame = _sorted_frame(y_true, treatment, uplift).reset_index(drop=True)
    treated = frame["treatment"] == 1
    control = ~treated

    cum_treated = treated.cumsum()
    cum_control = control.cumsum()
    cum_y_treated = (frame["target"] * treated.astype(int)).cumsum()
    cum_y_control = (frame["target"] * control.astype(int)).cumsum()

    control_scale = cum_treated / cum_control.replace(0, np.nan)
    qini = (cum_y_treated - control_scale.fillna(0.0) * cum_y_control).astype(float)

    return pd.DataFrame(
        {
            "fraction": (np.arange(len(frame)) + 1) / len(frame),
            "qini": qini,
        }
    )


def _qini_count_curve_arrays(
    y_true: Sequence[int] | np.ndarray,
    treatment: Sequence[int] | np.ndarray,
    uplift: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return count-space Qini curve arrays for normalized Qini ratios."""
    y, t, u = _validate_uplift_inputs(y_true, treatment, uplift)
    order = np.argsort(u, kind="mergesort")[::-1]
    y = y[order]
    t = t[order]
    u = u[order]

    y_control = y.copy()
    y_control[t == 1] = 0
    y_treatment = y.copy()
    y_treatment[t == 0] = 0

    distinct_value_indices = np.where(np.diff(u))[0]
    threshold_indices = np.r_[distinct_value_indices, u.size - 1]
    num_treatment = np.cumsum(t)[threshold_indices]
    num_all = threshold_indices + 1
    num_control = num_all - num_treatment
    treatment_responders = np.cumsum(y_treatment)[threshold_indices]
    control_responders = np.cumsum(y_control)[threshold_indices]
    curve_values = treatment_responders - control_responders * np.divide(
        num_treatment,
        num_control,
        out=np.zeros_like(num_treatment, dtype=float),
        where=num_control != 0,
    )

    if num_all.size == 0 or curve_values[0] != 0 or num_all[0] != 0:
        num_all = np.r_[0, num_all]
        curve_values = np.r_[0, curve_values]
    return num_all.astype(float), curve_values.astype(float)


def _perfect_qini_count_curve_arrays(
    y_true: Sequence[int] | np.ndarray,
    treatment: Sequence[int] | np.ndarray,
    *,
    negative_effect: bool,
) -> tuple[np.ndarray, np.ndarray]:
    y, t, _ = _validate_uplift_inputs(y_true, treatment, np.zeros(len(y_true)))
    if negative_effect:
        oracle_uplift = y * t - y * (1 - t)
        return _qini_count_curve_arrays(y, t, oracle_uplift)

    treated = t == 1
    control = ~treated
    random_ratio = y[treated].sum() - treated.sum() * y[control].sum() / control.sum()
    return (
        np.array([0.0, float(random_ratio), float(len(y))]),
        np.array([0.0, float(random_ratio), float(random_ratio)]),
    )


def uplift_curve_data(
    y_true: Sequence[int] | np.ndarray,
    treatment: Sequence[int] | np.ndarray,
    uplift: Sequence[float] | np.ndarray,
) -> pd.DataFrame:
    """Return cumulative treatment-control response-rate differences."""
    frame = _sorted_frame(y_true, treatment, uplift).reset_index(drop=True)
    treated = frame["treatment"] == 1
    control = ~treated

    cum_treated = treated.cumsum()
    cum_control = control.cumsum()
    cum_y_treated = (frame["target"] * treated.astype(int)).cumsum()
    cum_y_control = (frame["target"] * control.astype(int)).cumsum()

    treated_rate = cum_y_treated / cum_treated.replace(0, np.nan)
    control_rate = cum_y_control / cum_control.replace(0, np.nan)
    uplift_rate = (treated_rate - control_rate).fillna(0.0).astype(float)

    return pd.DataFrame(
        {
            "fraction": (np.arange(len(frame)) + 1) / len(frame),
            "uplift": uplift_rate,
        }
    )


def qini_auc_score(
    y_true: Sequence[int] | np.ndarray,
    treatment: Sequence[int] | np.ndarray,
    uplift: Sequence[float] | np.ndarray,
) -> float:
    """Area under the cumulative Qini curve."""
    curve = qini_curve_data(y_true, treatment, uplift)
    return round(_trapz_compatible(curve["qini"], curve["fraction"]), 6)


def normalized_qini_auc_score(
    y_true: Sequence[int] | np.ndarray,
    treatment: Sequence[int] | np.ndarray,
    uplift: Sequence[float] | np.ndarray,
    *,
    negative_effect: bool = False,
) -> float:
    """Return normalized Qini as actual area above baseline divided by oracle area.

    The default oracle excludes negative treatment effects, matching the
    report-facing normalized Qini used by the BT5153 benchmark notes.
    """
    x_actual, y_actual = _qini_count_curve_arrays(y_true, treatment, uplift)
    x_perfect, y_perfect = _perfect_qini_count_curve_arrays(
        y_true,
        treatment,
        negative_effect=negative_effect,
    )
    x_baseline = pd.Series([0.0, float(x_perfect[-1])])
    y_baseline = pd.Series([0.0, float(y_perfect[-1])])
    baseline_area = _trapz_compatible(y_baseline, x_baseline)
    perfect_area = _trapz_compatible(pd.Series(y_perfect), pd.Series(x_perfect))
    actual_area = _trapz_compatible(pd.Series(y_actual), pd.Series(x_actual))
    denominator = perfect_area - baseline_area
    if denominator == 0:
        return 0.0
    return round(float((actual_area - baseline_area) / denominator), 6)


def uplift_auc_score(
    y_true: Sequence[int] | np.ndarray,
    treatment: Sequence[int] | np.ndarray,
    uplift: Sequence[float] | np.ndarray,
) -> float:
    """Area under the cumulative uplift-rate curve."""
    curve = uplift_curve_data(y_true, treatment, uplift)
    return round(_trapz_compatible(curve["uplift"], curve["fraction"]), 6)


def uplift_at_k(
    y_true: Sequence[int] | np.ndarray,
    treatment: Sequence[int] | np.ndarray,
    uplift: Sequence[float] | np.ndarray,
    *,
    k: float,
) -> float:
    """Treatment-control response-rate difference among top-k ranked rows.

    Returns NaN when the top-k slice has zero treated or zero control rows: the
    uplift estimate is undefined, not zero. Callers should handle NaN explicitly
    rather than treating it as a value of 0.
    """
    if k <= 0 or k > 1:
        raise ValueError("k must be in (0, 1]")
    frame = _sorted_frame(y_true, treatment, uplift)
    top_n = max(1, int(np.ceil(len(frame) * k)))
    top = frame.head(top_n)
    treated = top[top["treatment"] == 1]
    control = top[top["treatment"] == 0]
    if treated.empty or control.empty:
        return float("nan")
    return round(float(treated["target"].mean() - control["target"].mean()), 6)


def decile_table(
    y_true: Sequence[int] | np.ndarray,
    treatment: Sequence[int] | np.ndarray,
    uplift: Sequence[float] | np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Build a ranked-bin table with response rates and observed uplift."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    frame = _sorted_frame(y_true, treatment, uplift).reset_index(drop=True)
    bins = np.array_split(frame.index.to_numpy(), min(n_bins, len(frame)))
    rows = []
    for i, idx in enumerate(bins, start=1):
        part = frame.loc[idx]
        treated = part[part["treatment"] == 1]
        control = part[part["treatment"] == 0]
        treated_rate = float(treated["target"].mean()) if not treated.empty else 0.0
        control_rate = float(control["target"].mean()) if not control.empty else 0.0
        rows.append(
            {
                "bin": i,
                "n": int(len(part)),
                "treated_n": int(len(treated)),
                "control_n": int(len(control)),
                "treated_response_rate": round(treated_rate, 6),
                "control_response_rate": round(control_rate, 6),
                "uplift": round(treated_rate - control_rate, 6),
                "avg_predicted_uplift": round(float(part["uplift"].mean()), 6),
            }
        )
    return pd.DataFrame(rows)


def policy_gain_by_cutoff(
    y_true: Sequence[int] | np.ndarray,
    treatment: Sequence[int] | np.ndarray,
    uplift: Sequence[float] | np.ndarray,
    policy: UpliftEvaluationPolicy,
) -> Dict[str, float]:
    """Estimate simple policy gain by cutoff and configured cost scenario.

    When the underlying uplift_at_k is undefined (NaN) for a cutoff, the gain
    for every cost scenario at that cutoff is also NaN: there is no honest
    way to claim a gain when the treatment-control comparison is undefined.
    """
    gains: Dict[str, float] = {}
    conversion_value = 1.0 if policy.conversion_value is None else policy.conversion_value
    n_rows = len(_as_1d_array(y_true, "y_true"))
    for cutoff in policy.cutoff_grid:
        observed_uplift = uplift_at_k(y_true, treatment, uplift, k=cutoff)
        n_contacted = int(np.ceil(n_rows * cutoff))
        cutoff_pct = int(round(cutoff * 100))
        for scenario, communication_cost in policy.cost_scenarios.items():
            key = f"top_{cutoff_pct}pct_{scenario}"
            if np.isnan(observed_uplift):
                gains[key] = float("nan")
                continue
            gain = observed_uplift * n_contacted * conversion_value
            gain -= n_contacted * communication_cost
            gains[key] = round(float(gain), 6)
    return gains


def evaluate_uplift_predictions(
    y_true: Sequence[int] | np.ndarray,
    treatment: Sequence[int] | np.ndarray,
    uplift: Sequence[float] | np.ndarray,
    policy: UpliftEvaluationPolicy,
) -> UpliftMetricResult:
    """Compute all first-pass uplift metrics and tabular artifacts."""
    qini_curve = qini_curve_data(y_true, treatment, uplift)
    uplift_curve = uplift_curve_data(y_true, treatment, uplift)
    at_k = {
        f"top_{int(round(cutoff * 100))}pct": uplift_at_k(
            y_true,
            treatment,
            uplift,
            k=cutoff,
        )
        for cutoff in policy.cutoff_grid
    }
    return UpliftMetricResult(
        qini_auc=qini_auc_score(y_true, treatment, uplift),
        uplift_auc=uplift_auc_score(y_true, treatment, uplift),
        uplift_at_k=at_k,
        policy_gain=policy_gain_by_cutoff(y_true, treatment, uplift, policy),
        decile_table=decile_table(y_true, treatment, uplift),
        qini_curve=qini_curve,
        uplift_curve=uplift_curve,
    )
