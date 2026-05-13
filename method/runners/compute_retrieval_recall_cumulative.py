#!/usr/bin/env python3
"""Cumulative budget split for retrieval recall.

For each K, three configurations holding the total pool size fixed at ≤ K:

  mag    : K  magnitude only
  +ons   : K/2 magnitude + K/2 earliest-onset    (deduped, ≤ K total)
  +stc   : K/3 magnitude + K/3 onset + K/3 stc   (deduped, ≤ K total)

This shows the marginal benefit of including each additional signal under a
fixed budget, so recall improvements are not driven by enlarging the pool.

Allocation (floor with remainder distributed mag → ons → stc):

  K = 5   mag: 5+0+0     +ons: 3+2+0     +stc: 2+2+1
  K = 10  mag: 10+0+0    +ons: 5+5+0     +stc: 4+3+3
  K = 15  mag: 15+0+0    +ons: 8+7+0     +stc: 5+5+5

Output: method/results/retrieval_recall_cumulative.csv
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

BHNP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BHNP_ROOT))

import numpy as np
import pandas as pd

from method.algorithms.rca.candidate_selection import (
    score_metrics, detect_earliest_onset, detect_state_changes,
)
from method.datasets.hvac import HVACDataset
from method.datasets.swat import SWaTDataset
from method.datasets.wadi import WADIDataset
from method.datasets.rcaeval_dataset import RCAEvalDataset


def _split_budget_across_signals(k: int, n_signals: int) -> list[int]:
    """floor(k/n_signals) base + remainder distributed to first signals.

    n_signals=1 → [k]
    n_signals=2 → [ceil(k/2), floor(k/2)]
    n_signals=3 → [k_mag, k_ons, k_stc] with k_mag ≥ k_ons ≥ k_stc, sum=k
    """
    base = k // n_signals
    rem = k % n_signals
    return [base + (1 if i < rem else 0) for i in range(n_signals)]


def _service_of(metric: str) -> str:
    return metric.split("_")[0].replace("-db", "")


def _service_truth(truth: str) -> str:
    return "_".join(truth.split("_")[:-1])


def _hit(truth_list, candidate_list, *, service_level: bool) -> int:
    if service_level:
        cand_svcs = []
        seen = set()
        for c in candidate_list:
            s = _service_of(c)
            if s not in seen:
                cand_svcs.append(s); seen.add(s)
        truth_svcs = [_service_truth(t) for t in truth_list]
        return int(any(t in cand_svcs for t in truth_svcs))
    return int(any(t in candidate_list for t in truth_list))


def _build_pool_cumulative(scenario, k: int, n_signals: int):
    """Cumulative pool: split K across the first `n_signals` signals."""
    data = scenario.data.ffill().fillna(0)
    diag = int(scenario.diagnosis_time)
    normal = data.iloc[:diag]
    anomal = data.iloc[diag:]

    splits = _split_budget_across_signals(k, n_signals)
    # Always allocate in order: magnitude, onset, state-change
    k_mag = splits[0] if n_signals >= 1 else 0
    k_ons = splits[1] if n_signals >= 2 else 0
    k_stc = splits[2] if n_signals >= 3 else 0

    scored = score_metrics(scenario)
    pool = []
    seen = set()
    for m, _ in scored[:k_mag]:
        if m not in seen:
            pool.append(m); seen.add(m)
    if k_ons > 0:
        for m, _ in detect_earliest_onset(normal, anomal, z_low=1.5)[:k_ons]:
            if m not in seen:
                pool.append(m); seen.add(m)
    if k_stc > 0:
        for m, _ in detect_state_changes(normal, anomal)[:k_stc]:
            if m not in seen:
                pool.append(m); seen.add(m)
    return pool, (k_mag, k_ons, k_stc)


def _eval(scenarios, *, service_level=False, K_list=(5, 10, 15)):
    rows = []
    for K in K_list:
        row = {"K": K, "n": len(scenarios)}
        for n_sig, label in [(1, "mag"), (2, "+ons"), (3, "+stc")]:
            hits = 0
            sizes = []
            split = _split_budget_across_signals(K, n_sig)
            split_str = "+".join(str(x) for x in split + [0] * (3 - n_sig))
            for sc in scenarios:
                pool, _ = _build_pool_cumulative(sc, K, n_sig)
                sizes.append(len(pool))
                hits += _hit(sc.ground_truth_causes, pool, service_level=service_level)
            row[f"{label}_recall"] = round(hits / max(len(scenarios), 1), 4)
            row[f"{label}_split"] = split_str
            row[f"{label}_pool"]  = round(float(np.mean(sizes)), 2)
        rows.append(row)
    return rows


def main():
    print("Loading datasets…")
    rcaeval_all = RCAEvalDataset(suites=["RE1-OB", "RE1-SS", "RE1-TT"]).load_fault_scenarios()
    re1ob = [s for s in rcaeval_all if s.metadata.get("suite") == "RE1-OB"]
    re1ss = [s for s in rcaeval_all if s.metadata.get("suite") == "RE1-SS"]
    re1tt = [s for s in rcaeval_all if s.metadata.get("suite") == "RE1-TT"]
    datasets = [
        ("RE1-OB", re1ob, True),  # service-level (RCAEval ASE'24 protocol)
        ("RE1-SS", re1ss, True),
        ("RE1-TT", re1tt, True),
        ("WADI",   WADIDataset().load_fault_scenarios(), False),
        ("SWaT",   SWaTDataset().load_fault_scenarios(), False),
        ("HVAC",   HVACDataset().load_fault_scenarios(), False),
    ]

    out = []
    for name, scs, svc in datasets:
        print(f"\n=== {name} (n={len(scs)}) — cumulative split ===")
        for r in _eval(scs, service_level=svc):
            r["dataset"] = name
            print(
                f"  K={r['K']:>2}  "
                f"mag {r['mag_split']:>8s} = {r['mag_recall']:.3f} (pool {r['mag_pool']})   "
                f"+ons {r['+ons_split']:>8s} = {r['+ons_recall']:.3f} (pool {r['+ons_pool']})   "
                f"+stc {r['+stc_split']:>8s} = {r['+stc_recall']:.3f} (pool {r['+stc_pool']})"
            )
            out.append(r)

    df = pd.DataFrame(out)
    out_dir = BHNP_ROOT / "method" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "retrieval_recall_cumulative.csv"
    df.to_csv(p, index=False)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
