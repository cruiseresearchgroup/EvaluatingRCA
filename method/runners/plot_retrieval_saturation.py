#!/usr/bin/env python3
"""Saturation curves: recall@K vs K, per signal, per dataset.

Each signal gets its own full K (no shared budget) so the curves show
how quickly each retriever covers the ground-truth cause as the
candidate budget grows.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

BHNP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BHNP_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from method.algorithms.rca.candidate_selection import (
    score_metrics, detect_earliest_onset, detect_state_changes,
)
from method.datasets.hvac import HVACDataset
from method.datasets.swat import SWaTDataset
from method.datasets.wadi import WADIDataset
from method.datasets.rcaeval_dataset import RCAEvalDataset


def _service_of(metric: str) -> str:
    return metric.split("_")[0].replace("-db", "")


def _service_truth(truth: str) -> str:
    return "_".join(truth.split("_")[:-1])


def _hit(truth_list, candidate_list, *, service_level: bool) -> int:
    if service_level:
        cand_svcs, seen = [], set()
        for c in candidate_list:
            s = _service_of(c)
            if s not in seen:
                cand_svcs.append(s); seen.add(s)
        truth_svcs = [_service_truth(t) for t in truth_list]
        return int(any(t in cand_svcs for t in truth_svcs))
    return int(any(t in candidate_list for t in truth_list))


def _signal_lists(scenario):
    """Return full-length ranked lists per signal (no truncation)."""
    data = scenario.data.ffill().fillna(0)
    diag = int(scenario.diagnosis_time)
    normal = data.iloc[:diag]
    anomal = data.iloc[diag:]
    mag = [m for m, _ in score_metrics(scenario)]
    ons = [m for m, _ in detect_earliest_onset(normal, anomal, z_low=1.5)]
    stc = [m for m, _ in detect_state_changes(normal, anomal)]
    return mag, ons, stc


def _curves(scenarios, K_list, *, service_level: bool):
    rows = []
    for sc in scenarios:
        mag, ons, stc = _signal_lists(sc)
        for K in K_list:
            row = {
                "K": K,
                "mag": _hit(sc.ground_truth_causes, mag[:K], service_level=service_level),
                "ons": _hit(sc.ground_truth_causes, ons[:K], service_level=service_level),
                "stc": _hit(sc.ground_truth_causes, stc[:K], service_level=service_level),
                "union": _hit(
                    sc.ground_truth_causes,
                    list(dict.fromkeys(mag[:K] + ons[:K] + stc[:K])),
                    service_level=service_level,
                ),
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    return df.groupby("K").mean().reset_index()


def main():
    print("Loading datasets…")
    rcaeval_all = RCAEvalDataset(
        suites=["RE1-OB", "RE1-SS", "RE1-TT"],
    ).load_fault_scenarios()
    re1ob = [s for s in rcaeval_all if s.metadata.get("suite") == "RE1-OB"]
    re1ss = [s for s in rcaeval_all if s.metadata.get("suite") == "RE1-SS"]
    re1tt = [s for s in rcaeval_all if s.metadata.get("suite") == "RE1-TT"]

    datasets = [
        ("WADI",   WADIDataset().load_fault_scenarios(), False, 79),
        ("SWaT",   SWaTDataset().load_fault_scenarios(), False, 51),
        ("HVAC",   HVACDataset().load_fault_scenarios(), False, None),
        ("RE1-OB", re1ob, True, None),
        ("RE1-SS", re1ss, True, None),
        ("RE1-TT", re1tt, True, None),
    ]

    K_list = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50]

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 14,
    })

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharey=True)
    axes = axes.flatten()

    saved = []
    colors = {"mag": "#1f77b4", "ons": "#d62728", "stc": "#2ca02c"}
    styles = {"mag": "-", "ons": "-", "stc": "-"}

    for ax, (name, scs, svc, _p) in zip(axes, datasets):
        print(f"  {name} (n={len(scs)}) …")
        df = _curves(scs, K_list, service_level=svc)
        df["dataset"] = name
        saved.append(df)
        for col in ("mag", "ons", "stc"):
            ax.plot(df["K"], df[col], color=colors[col], linestyle=styles[col],
                    marker="o", markersize=5, linewidth=2.2, label=col)
        ax.set_title(f"{name} (n={len(scs)})")
        ax.set_xlabel("K")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.axvline(15, color="grey", linestyle=":", linewidth=1.0, alpha=0.7)

    axes[0].set_ylabel("recall@K")
    axes[3].set_ylabel("recall@K")
    axes[0].legend(loc="lower right", framealpha=0.95)
    plt.tight_layout()

    out_dir = BHNP_ROOT / "method" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "retrieval_saturation.png"
    out_pdf = out_dir / "retrieval_saturation.pdf"
    out_csv = out_dir / "retrieval_saturation.csv"
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    pd.concat(saved, ignore_index=True).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_png}\nSaved: {out_pdf}\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
