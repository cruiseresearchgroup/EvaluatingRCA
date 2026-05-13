"""Hybrid candidate selection for LLM-based RCA.

Merges three signals to ensure low-magnitude but causally important
metrics are not dropped from the LLM's view:

  1. Top-N by RobustScaler z-score (standard Baro-style magnitude ranking) 50:50 vs standard scaler (TODO: Validate)
  2. Earliest onset at a low z-threshold (catches early but small signals)
  3. State/discrete changes (catches binary flips: valve open→closed, pump on→off)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from method.datasets.base import FaultScenario


def score_metrics(scenario: FaultScenario) -> list[tuple[str, float]]:
    """Score each metric by max RobustScaler z-score (fault vs baseline).

    Returns list of (metric_name, score) sorted descending.
    """
    data = scenario.data.ffill().fillna(0)
    diag = int(scenario.diagnosis_time)
    normal = data.iloc[:diag]
    anomal = data.iloc[diag:]

    if normal.empty or anomal.empty:
        return [(c, 0.0) for c in data.columns]

    scores = []
    for col in data.columns:
        a = normal[col].to_numpy(dtype=float).reshape(-1, 1)
        b = anomal[col].to_numpy(dtype=float).reshape(-1, 1)
        try:
            scaler = RobustScaler().fit(a)
            zscores = scaler.transform(b)[:, 0]
            scores.append((col, float(np.max(zscores))))
        except Exception:
            scores.append((col, 0.0))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def detect_state_changes(
    normal: pd.DataFrame, anomal: pd.DataFrame, tol: float = 0.01,
) -> list[tuple[str, int]]:
    """Find metrics that flipped between two discrete states (binary/ternary).

    Returns list of (metric, first_change_row) for metrics where:
      - The baseline has ≤5 unique values (discrete signal)
      - The fault window has a different dominant value
    """
    changes = []
    for col in anomal.columns:
        if col not in normal.columns:
            continue
        n_unique_base = normal[col].nunique()
        n_unique_fault = anomal[col].nunique()
        if n_unique_base > 5 and n_unique_fault > 5: # only consider discrete signals with nunique values ≤ 5
            continue

        base_mode = normal[col].mode().iloc[0] if not normal[col].mode().empty else None
        if base_mode is None:
            continue

        fault_mode = anomal[col].mode().iloc[0] if not anomal[col].mode().empty else None
        if fault_mode is None or abs(fault_mode - base_mode) < tol:
            continue

        diff = anomal[col][abs(anomal[col] - base_mode) > tol]
        if not diff.empty:
            changes.append((col, int(diff.index[0])))

    changes.sort(key=lambda x: x[1])
    return changes


def detect_earliest_onset(
    normal: pd.DataFrame, anomal: pd.DataFrame, z_low: float = 1.5,
) -> list[tuple[str, int]]:
    """Find metrics with early anomaly onset even at a low z-threshold.

    Returns list of (metric, first_onset_row) sorted by onset time.
    """
    b_mean = normal.mean()
    b_std = normal.std()

    onsets = []
    for col in anomal.columns:
        if col not in normal.columns:
            continue
        std_val = b_std.get(col, 0.0)
        if pd.isna(std_val) or std_val == 0:
            bval = b_mean.get(col, 0.0)
            if pd.isna(bval):
                continue
            changed = anomal[col][(anomal[col] - bval).abs() > 1e-4]
            if not changed.empty:
                onsets.append((col, int(changed.index[0])))
        else:
            z = ((anomal[col] - b_mean[col]) / std_val).abs()
            exceeds = z[z >= z_low]
            if not exceeds.empty:
                onsets.append((col, int(exceeds.index[0])))

    onsets.sort(key=lambda x: x[1])
    return onsets


def _split_k_three(k: int) -> tuple[int, int, int]:
    """Allocate K across three signals: mag ≥ ons ≥ stc, sum=K.

    Convention: floor(K/3) base, distribute remainder mag→ons→stc.
      K=5  → 2 mag + 2 ons + 1 stc
      K=10 → 4 mag + 3 ons + 3 stc
      K=15 → 5 mag + 5 ons + 5 stc
    """
    base = k // 3
    rem = k % 3
    k_mag = base + (1 if rem >= 1 else 0)
    k_ons = base + (1 if rem >= 2 else 0)
    k_stc = base
    return k_mag, k_ons, k_stc


def select_candidates_balanced(
    scenario: FaultScenario,
    k: int = 15,
    k_mag: int | None = None,
    k_ons: int | None = None,
    k_stc: int | None = None,
) -> list[tuple[str, float, str]]:
    """K-budget balanced candidate selection: each signal contributes K/3 items.

    Returns deduped union; pool size ≤ k (smaller when signals overlap).
    Order preserved: magnitude → onset → state-change.

    See `_split_k_three` for the K/3 allocation convention.
    """
    if k_mag is None or k_ons is None or k_stc is None:
        k_mag, k_ons, k_stc = _split_k_three(k)
    return select_candidates(
        scenario,
        top_n_score=k_mag,
        top_n_early=k_ons,
        top_n_state=k_stc,
    )


def select_candidates(
    scenario: FaultScenario,
    top_n_score: int = 15,
    top_n_early: int = 5,
    top_n_state: int = 5,
) -> list[tuple[str, float, str]]:
    """Hybrid candidate selection: merge three signals.

    Returns list of (metric, z_score, selection_reason) with no duplicates.
    Order: score-based first, then early-onset, then state-change.
    """
    data = scenario.data.ffill().fillna(0)
    diag = int(scenario.diagnosis_time)
    normal = data.iloc[:diag]
    anomal = data.iloc[diag:]

    scored = score_metrics(scenario)
    score_dict = dict(scored)

    selected = []
    seen = set()

    for metric, z in scored[:top_n_score]:
        selected.append((metric, z, "z-score"))
        seen.add(metric)

    early = detect_earliest_onset(normal, anomal, z_low=1.5)
    for metric, _ in early[:top_n_early]:
        if metric not in seen:
            selected.append((metric, score_dict.get(metric, 0.0), "early-onset"))
            seen.add(metric)

    state_changes = detect_state_changes(normal, anomal)
    for metric, _ in state_changes[:top_n_state]:
        if metric not in seen:
            selected.append((metric, score_dict.get(metric, 0.0), "state-change"))
            seen.add(metric)

    return selected


def build_anomaly_summary(
    scenario: FaultScenario, top_n: int = 15, time_unit: str = "s",
) -> str:
    """Build a text summary of candidate metrics for the LLM prompt.

    Renders one line per candidate metric containing magnitude (z-score),
    onset offset, and before/after means — the evidence format used in
    the paper's LLM-reranker rows.
    """
    data = scenario.data.ffill().fillna(0)
    diag = int(scenario.diagnosis_time)
    normal = data.iloc[:diag]
    anomal = data.iloc[diag:]

    b_mean = normal.mean()
    b_std = normal.std()

    candidates = select_candidates(scenario, top_n_score=top_n)

    lines = []
    for rank, (metric, z, _reason) in enumerate(candidates, 1):
        mean_before = b_mean.get(metric, 0)
        std_before = b_std.get(metric, 0)
        mean_after = anomal[metric].mean() if metric in anomal else 0

        if std_before > 0:
            zvals = ((anomal[metric] - mean_before) / std_before).abs()
            exceeds = zvals[zvals >= 1.5]
            offset = int(exceeds.index[0]) - diag if not exceeds.empty else "?"
        else:
            bval = mean_before
            changed = anomal[metric][(anomal[metric] - bval).abs() > 1e-4]
            offset = int(changed.index[0]) - diag if not changed.empty else "?"

        delta = mean_after - mean_before
        pct = (delta / mean_before * 100) if mean_before != 0 else 0

        lines.append(
            f"  {rank}. {metric}  |  z-score={z:.1f}  |  +{offset}{time_unit}  |  "
            f"before={mean_before:.3f} → after={mean_after:.3f} "
            f"(Δ={delta:+.3f}, {pct:+.1f}%)"
        )

    return "\n".join(lines)
