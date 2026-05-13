#!/usr/bin/env python3
"""Baseline rows for Tables 5 / 6 — one runner, four datasets.

Runs the paper's statistical and graph-based baselines:

  Statistical:    BARO, RCD, EpsilonDiagnosis  (no graph required)
  Graph-based:    PageRank, RandomWalk, CIRCA  × PC, FCI causal-discovery learners

Per-scenario PC / FCI graphs are cached under
``method/datasets/<ds>/graph_cache_per_scenario/`` so that re-running this
script reads pre-fitted graphs instead of re-learning them (cached graphs
ship with the repo; see DATA.md).

Usage
-----
    python method/runners/run_baseline.py --dataset wadi
    python method/runners/run_baseline.py --dataset swat
    python method/runners/run_baseline.py --dataset hvac
    python method/runners/run_baseline.py --dataset rcaeval                # all suites
    python method/runners/run_baseline.py --dataset rcaeval --suite RE1-OB # one suite
    python method/runners/run_baseline.py --dataset swat --algos Baro RCD

Replaces the per-dataset ``run_{wadi,swat,hvac,rcaeval}.py`` and
``run_{wadi,hvac}_baselines.py`` scripts that previously lived here.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

BHNP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BHNP_ROOT))

from dotenv import load_dotenv
load_dotenv(BHNP_ROOT / ".env")

import numpy as np
import pandas as pd

from method.algorithms.cd.fci import FCIAdapter
from method.algorithms.cd.pc import PCAdapter
from method.algorithms.cd.per_scenario import fit_graph_per_scenario
from method.algorithms.rca.baro import BaroAdapter
from method.algorithms.rca.circa import CIRCAAdapter
from method.algorithms.rca.epsilon_diagnosis import EpsilonDiagnosisAdapter
from method.algorithms.rca.pagerank import PageRankAdapter
from method.algorithms.rca.random_walk import RandomWalkAdapter
from method.algorithms.rca.rcd import RCDAdapter
from method.datasets.base import FaultScenario
from method.datasets.rcaeval_dataset import SUITES as RCAEVAL_SUITES
from method.evaluation.metrics import avg_at_k, top_at_k
from method.runners._common import compute_metrics, merge_aggregate
from method.runners._datasets import get as get_dataset


# ---------------------------------------------------------------------------
# Graph-algorithm wrapper: fits CD graph per scenario, then runs the head adapter
# ---------------------------------------------------------------------------

class _GraphAlgoWrapper:
    """Fit a per-scenario causal graph, then call a graph-based RCA head."""

    def __init__(self, head_adapter, cd_cls, dataset_name: str, cache_dir: Path,
                 cd_kwargs: dict | None = None):
        self.head = head_adapter
        self.cd_cls = cd_cls
        self.cd_kwargs = cd_kwargs or {}
        self.cd_name = cd_cls.__name__.replace("Adapter", "")
        self.dataset = dataset_name
        self.cache_dir = cache_dir

    def predict(self, scenario):
        adapter = self.cd_cls(**self.cd_kwargs)
        graph = fit_graph_per_scenario(
            adapter, scenario, dataset=self.dataset,
            cd_name=self.cd_name, cache_dir=self.cache_dir,
        )
        if graph is None or graph.empty:
            return list(scenario.alarm_nodes)
        return self.head.predict(scenario, graph=graph)


def _algo_factories(cfg) -> dict[str, callable]:
    """Build the (algo_name → factory) dict for one dataset."""
    return {
        "Baro":              BaroAdapter,
        "RCD":               lambda: RCDAdapter(patch=cfg.patch),
        "EpsilonDiagnosis":  lambda: EpsilonDiagnosisAdapter(patch=cfg.patch),
        "PageRank (PC)":     lambda: _wrap(PageRankAdapter,   PCAdapter,  cfg),
        "PageRank (FCI)":    lambda: _wrap(PageRankAdapter,   FCIAdapter, cfg),
        "RandomWalk (PC)":   lambda: _wrap(RandomWalkAdapter, PCAdapter,  cfg),
        "RandomWalk (FCI)":  lambda: _wrap(RandomWalkAdapter, FCIAdapter, cfg),
        "CIRCA (PC)":        lambda: _wrap(CIRCAAdapter,      PCAdapter,  cfg),
        "CIRCA (FCI)":       lambda: _wrap(CIRCAAdapter,      FCIAdapter, cfg),
    }


def _wrap(head_cls, cd_cls, cfg):
    return _GraphAlgoWrapper(
        head_adapter=head_cls(), cd_cls=cd_cls,
        dataset_name=cfg.name, cache_dir=cfg.graph_cache_dir,
    )


# ---------------------------------------------------------------------------
# Service-level evaluation helper (RCAEval only)
# ---------------------------------------------------------------------------

def _service_ranks(pred: list[str]) -> list[str]:
    """Convert metric-level predictions to service-level (strip metric suffix, dedup).

    Matches RCAEval ASE'24 main-ase.py: ``Node(x.split("_")[0].replace("-db", ""), ...)``.
    """
    seen: list[str] = []
    for p in pred:
        if not p or "_" not in p:
            continue
        svc = p.split("_")[0].replace("-db", "")
        if svc not in seen:
            seen.append(svc)
    return seen


_FAULT_METRIC_MAP = {
    "cpu": "cpu", "mem": "mem", "delay": "latency",
    "loss": "latency", "disk": "diskio",
}


def _service_truths(scenarios: list[FaultScenario]) -> list[list[str]]:
    """Strip ``_<fault>`` suffix from each ground-truth label → service name."""
    out = []
    for sc in scenarios:
        out.append(["_".join(t.split("_")[:-1]).replace("-db", "")
                    for t in sc.ground_truth_causes])
    return out


def _metric_truths(scenarios: list[FaultScenario]) -> list[list[str]]:
    """Map ``{service}_{fault}`` ground truth → ``{service}_{metric}`` (RCAEval protocol)."""
    out = []
    for sc in scenarios:
        mapped = []
        for t in sc.ground_truth_causes:
            parts = t.split("_")
            service = "_".join(parts[:-1])
            fault = parts[-1]
            metric = _FAULT_METRIC_MAP.get(fault, fault)
            mapped.append(f"{service}_{metric}")
        out.append(mapped)
    return out


def _compute_rcaeval_metrics(
    scenarios: list[FaultScenario],
    predictions: list[list[str]],
    label: str,
    group: str,
) -> dict:
    """Compute both service-level (``top@k_svc``) and metric-level (``top@k_met``)."""
    svc_truths = _service_truths(scenarios)
    met_truths = _metric_truths(scenarios)
    svc_preds = [_service_ranks(p) for p in predictions]

    out = {"algorithm": label, "group": group, "n": len(scenarios)}
    for k in (1, 3, 5):
        out[f"top@{k}_svc"] = round(float(np.mean(
            [top_at_k(t, p, k) for t, p in zip(svc_truths, svc_preds)]
        )), 4)
        out[f"top@{k}_met"] = round(float(np.mean(
            [top_at_k(t, p, k) for t, p in zip(met_truths, predictions)]
        )), 4)
    out["avg@5_svc"] = round(avg_at_k(svc_truths, svc_preds, 5), 4)
    out["avg@5_met"] = round(avg_at_k(met_truths, predictions, 5), 4)
    return out


def _merge_rcaeval(path: Path, new_row: dict) -> None:
    """RCAEval upsert keyed by (algorithm, group) instead of algorithm alone."""
    key = (new_row["algorithm"], new_row["group"])
    if path.exists():
        df = pd.read_csv(path)
        df = df[~((df["algorithm"] == key[0]) & (df["group"] == key[1]))]
    else:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True, sort=False)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Per-(dataset, suite?) run loop
# ---------------------------------------------------------------------------

def run_dataset(dataset_name: str, algos: list[str] | None, suite: str | None,
                save_dir: Path) -> None:
    cfg = get_dataset(dataset_name)
    algo_factories = _algo_factories(cfg)
    if algos is None:
        algos = list(algo_factories.keys())
    agg_path = save_dir / cfg.output_csv

    if cfg.name == "rcaeval":
        suites = [suite] if suite else list(RCAEVAL_SUITES)
        per_suite_preds: dict[str, dict] = {}     # suite → {algo → preds}
        per_suite_scenarios: dict[str, list] = {}
        for s in suites:
            ds = cfg.loader(suites=[s])
            scenarios = ds.load_fault_scenarios()
            per_suite_scenarios[s] = scenarios
            per_suite_preds[s] = {}
            print(f"\n## RCAEval {s} — {len(scenarios)} scenarios")
            for algo_name in algos:
                adapter = algo_factories[algo_name]()
                print(f"=== {algo_name} ({s}) ===")
                t0 = time.time()
                preds = []
                for sc in scenarios:
                    try:
                        preds.append(adapter.predict(sc))
                    except Exception as e:
                        print(f"  [{sc.scenario_id}] error: {e}")
                        preds.append(sc.alarm_nodes[:])
                per_suite_preds[s][algo_name] = preds
                row = _compute_rcaeval_metrics(scenarios, preds, algo_name, group=s)
                print(
                    f"  done in {time.time() - t0:.1f}s | "
                    f"top@1_svc={row['top@1_svc']:.4f}  top@5_svc={row['top@5_svc']:.4f}  "
                    f"avg@5_svc={row['avg@5_svc']:.4f}"
                )
                _merge_rcaeval(agg_path, row)
        # ALL-suite aggregate
        for algo_name in algos:
            all_scens, all_preds = [], []
            for s in suites:
                all_scens.extend(per_suite_scenarios[s])
                all_preds.extend(per_suite_preds[s][algo_name])
            row = _compute_rcaeval_metrics(all_scens, all_preds, algo_name, group="ALL")
            print(
                f"  [ALL] {algo_name}  top@1_svc={row['top@1_svc']:.4f}  "
                f"avg@5_svc={row['avg@5_svc']:.4f}"
            )
            _merge_rcaeval(agg_path, row)
    else:
        # Single-pool datasets (WADI, SWaT, HVAC)
        print(f"Loading {cfg.name}…")
        scenarios = cfg.loader().load_fault_scenarios()
        print(f"  Loaded {len(scenarios)} scenarios\n")

        for algo_name in algos:
            adapter = algo_factories[algo_name]()
            print(f"=== {algo_name} ===")
            t0 = time.time()
            preds = []
            for sc in scenarios:
                try:
                    preds.append(adapter.predict(sc))
                except Exception as e:
                    print(f"  [{sc.scenario_id}] error: {e}")
                    preds.append(sc.alarm_nodes[:])
            metrics = compute_metrics(scenarios, preds, algo_name)
            print(
                f"  done in {time.time() - t0:.1f}s | "
                f"top@1={metrics['top@1']:.4f}  top@3={metrics['top@3']:.4f}  "
                f"top@5={metrics['top@5']:.4f}  avg@5={metrics['avg@5']:.4f}"
            )
            merge_aggregate(agg_path, metrics)

    print(f"\n✓ wrote {agg_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Baseline rows for Tables 5/6 — one runner, four datasets."
    )
    p.add_argument("--dataset", required=True,
                   choices=["wadi", "swat", "hvac", "rcaeval"])
    p.add_argument("--suite", default=None,
                   help="RCAEval only: which RE1 suite. Omit for all suites.")
    p.add_argument("--algos", nargs="+", default=None,
                   help="Subset of algorithms to run. Default: all paper algos.")
    p.add_argument("--save-dir", default=str(BHNP_ROOT / "method" / "results"))
    args = p.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    run_dataset(args.dataset, args.algos, args.suite, save_dir)


if __name__ == "__main__":
    main()
