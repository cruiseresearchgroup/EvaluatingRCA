#!/usr/bin/env python3
"""LLM-reranker rows for Tables 5 / 6 — one runner, four datasets.

Implements the paper's balanced K=15 (5 magnitude + 5 onset + 5 state-change)
candidate-selection policy + ``hybrid_clean`` summary mode + light-DK / no-DK
ablation. Each (run_idx, dataset, dk_level, scenario) call is independently
cached and Langfuse-traced.

Usage
-----
    python method/runners/run_llm_balanced.py --dataset wadi
    python method/runners/run_llm_balanced.py --dataset swat
    python method/runners/run_llm_balanced.py --dataset hvac
    python method/runners/run_llm_balanced.py --dataset rcaeval --suite RE1-OB
    python method/runners/run_llm_balanced.py --dataset rcaeval          # all suites

Replaces the per-dataset ``run_{wadi,swat,hvac,rcaeval}_llm_balanced.py``
scripts that previously lived in this directory.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

# Path bootstrap so the script can run directly as ``python path/to/file.py``.
BHNP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BHNP_ROOT))

from dotenv import load_dotenv
load_dotenv(BHNP_ROOT / ".env")

import numpy as np
import pandas as pd

from method.datasets.rcaeval_dataset import SUITES as RCAEVAL_SUITES
from method.runners._common import (
    DK_DOC_TAG, SELECTION_POLICY, SUMMARY_MODE, TOP_N,
    build_system_prompt, compute_metrics, llm_cache_dir, load_context,
    log_run_summary, make_client, merge_aggregate, predict_one_scenario,
    run_suffix,
)
from method.runners._datasets import (
    DATASETS, get as get_dataset,
    rcaeval_context_for_suite, rcaeval_domain_for_suite,
)

try:
    from langfuse import get_client
    import os as _os
    _LANGFUSE = get_client() if _os.environ.get("LANGFUSE_PUBLIC_KEY") else None
except Exception:
    _LANGFUSE = None


# ---------------------------------------------------------------------------
# Single (dataset, level, run) block
# ---------------------------------------------------------------------------

def run_one_block(
    *,
    dataset_name: str,
    scenarios: list,
    level: str,
    domain_phrase: str,
    domain_context: str,
    model: str,
    workers: int,
    run_idx: int,
    n_runs: int,
    temperature: float,
    run_name: str,
    session_id: str,
    dataset_root: Path,
    time_unit: str,
    agg_path: Path,
    client,
    suite: str | None = None,
) -> dict:
    """Run the LLM over one (dataset, DK level, run_idx) combination.

    Returns the aggregate metrics row (for the stability summary).
    """
    system_prompt = build_system_prompt(level, domain_phrase, domain_context)
    suf = run_suffix(run_idx, temperature)
    label_suffix = "" if not suf else f", run={run_idx}, t={temperature}"
    dk_label = "no DK" if level == "none" else f"DK={DK_DOC_TAG}"
    suite_tag = f"::suite={suite}" if suite else ""
    label = (
        f"LLM Ranking ({dk_label}, {SUMMARY_MODE}, {SELECTION_POLICY}"
        f"{label_suffix})"
    )

    cache_dir = llm_cache_dir(dataset_root, model, level, suf)
    if suite:
        cache_dir = cache_dir / suite

    print(f"\n=== {label} | DK chars: {len(domain_context)} "
          f"| n_scenarios={len(scenarios)}{suite_tag} ===")
    t0 = time.time()
    results: dict[str, list[str]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                predict_one_scenario, sc,
                dataset_name=dataset_name, time_unit=time_unit,
                system_prompt=system_prompt, level=level,
                model=model, client=client, cache_dir=cache_dir,
                run_name=run_name, session_id=session_id,
                run_idx=run_idx, n_runs=n_runs, temperature=temperature,
            ): sc for sc in scenarios
        }
        for fut in concurrent.futures.as_completed(futures):
            sc = futures[fut]
            try:
                sid, ranked, source = fut.result()
                results[sid] = ranked
                print(f"  [{sid}] {source}  top1={ranked[0] if ranked else '?'}")
            except Exception as e:
                print(f"  [{sc.scenario_id}] error: {e}")
                results[sc.scenario_id] = sc.alarm_nodes[:]
    print(f"  done in {time.time() - t0:.1f}s")

    preds = [results[sc.scenario_id] for sc in scenarios]
    metrics = compute_metrics(scenarios, preds, label)
    metrics["run_idx"] = run_idx
    metrics["temperature"] = temperature
    if suite:
        metrics["suite"] = suite
    print(
        f"  top@1={metrics['top@1']:.4f}  top@3={metrics['top@3']:.4f}  "
        f"top@5={metrics['top@5']:.4f}  avg@5={metrics['avg@5']:.4f}"
    )
    merge_aggregate(agg_path, metrics)
    print(f"  merged -> {agg_path}")

    log_run_summary(
        session_id=session_id, run_name=run_name,
        name_suffix=(
            f"{dataset_name}::dk={level}::dk_doc="
            f"{DK_DOC_TAG if level != 'none' else 'none'}"
            f"::{SUMMARY_MODE}::{SELECTION_POLICY}"
            f"::run={run_idx}::t={temperature}{suite_tag}"
        ),
        metrics=metrics,
        metadata={
            "dataset": dataset_name, "dk_level": level,
            "dk_doc": DK_DOC_TAG if level != "none" else None,
            "summary_mode": SUMMARY_MODE,
            "selection_policy": SELECTION_POLICY,
            "run_idx": run_idx, "n_runs": n_runs,
            "temperature": temperature, "model": model,
            "n_scenarios": len(scenarios),
            "dk_chars": len(domain_context),
            **({"suite": suite} if suite else {}),
        },
        tags=[
            f"dataset={dataset_name}", f"dk={level}",
            f"dk_doc={DK_DOC_TAG if level != 'none' else 'none'}",
            f"summary_mode={SUMMARY_MODE}",
            f"selection_policy={SELECTION_POLICY}",
            f"run_idx={run_idx}", f"n_runs={n_runs}",
            f"temperature={temperature}",
            *([f"suite={suite}"] if suite else []),
        ],
    )
    return metrics


# ---------------------------------------------------------------------------
# Stability summary across n_runs (paper Table 5/6 are the n_runs=3 mean)
# ---------------------------------------------------------------------------

def write_stability_summary(
    *, dataset_name: str, save_dir: Path,
    per_level_runs: dict[str, list[dict]], n_runs: int,
    temperature: float, model: str, session_id: str, run_name: str,
):
    if n_runs <= 1:
        return
    print(f"\n{'=' * 60}\n  Stability summary across {n_runs} runs\n{'=' * 60}")
    stability_path = save_dir / f"{dataset_name}_llm_stability.csv"
    rows = []
    for level, runs in per_level_runs.items():
        if not runs:
            continue
        row = {"dataset": dataset_name, "dk_level": level, "n_runs": len(runs),
               "temperature": temperature, "model": model,
               "selection_policy": SELECTION_POLICY,
               "summary_mode": SUMMARY_MODE}
        for k_label in ("top@1", "top@3", "top@5", "avg@5"):
            vals = [m[k_label] for m in runs]
            row[f"{k_label}_mean"] = round(float(np.mean(vals)), 4)
            row[f"{k_label}_std"] = round(
                float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4
            )
            row[f"{k_label}_runs"] = ",".join(f"{v:.4f}" for v in vals)
        rows.append(row)
        print(f"  {level}:  top@1 = {row['top@1_mean']:.4f} ± {row['top@1_std']:.4f}  "
              f"top@3 = {row['top@3_mean']:.4f} ± {row['top@3_std']:.4f}  "
              f"top@5 = {row['top@5_mean']:.4f} ± {row['top@5_std']:.4f}")
    if rows:
        df = pd.DataFrame(rows)
        if stability_path.exists():
            old = pd.read_csv(stability_path)
            df = pd.concat([old, df], ignore_index=True, sort=False)
        df.to_csv(stability_path, index=False)
        print(f"\n  stability stats -> {stability_path}")

    for row in rows:
        stab_metrics = {k: v for k, v in row.items()
                        if k.endswith(("_mean", "_std", "_runs"))}
        log_run_summary(
            session_id=session_id, run_name=run_name,
            name_suffix=(
                f"{dataset_name}::dk={row['dk_level']}::dk_doc="
                f"{DK_DOC_TAG if row['dk_level'] != 'none' else 'none'}"
                f"::{SUMMARY_MODE}::{SELECTION_POLICY}"
                f"::stability::n_runs={row['n_runs']}"
            ),
            metrics={"algorithm": "stability_summary", **stab_metrics},
            metadata={
                "dataset": dataset_name, "dk_level": row["dk_level"],
                "n_runs": row["n_runs"], "temperature": temperature,
                "model": model,
                "selection_policy": SELECTION_POLICY,
                "summary_mode": SUMMARY_MODE,
            },
            tags=[
                "stability", f"dataset={dataset_name}", f"dk={row['dk_level']}",
                f"n_runs={row['n_runs']}", f"temperature={temperature}",
                f"selection_policy={SELECTION_POLICY}",
            ],
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="LLM reranker rows for Tables 5/6 — one runner, four datasets."
    )
    p.add_argument("--dataset", required=True,
                   choices=sorted(DATASETS.keys()))
    p.add_argument("--suite", default=None,
                   help="RCAEval only: which RE1 suite to run. Omit for all.")
    p.add_argument("--model", default="openai/gpt-oss-120b")
    p.add_argument("--context", choices=["none", "light", "all"], default="all",
                   help="Domain-knowledge level. 'all' runs both none and light.")
    p.add_argument("--workers", type=int, default=8,
                   help="Concurrent LLM calls per DK level. Default 8.")
    p.add_argument("--n-runs", type=int, default=3,
                   help="Number of independent runs per (DK level, scenario).")
    p.add_argument("--run-idx", type=int, default=0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--save-dir", default=str(BHNP_ROOT / "method" / "results"))
    args = p.parse_args()

    cfg = get_dataset(args.dataset)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    agg_path = save_dir / cfg.output_csv

    # Load scenarios. RCAEval is per-suite; everything else is whole-dataset.
    print(f"Loading {cfg.name}…")
    if cfg.name == "rcaeval":
        suites = [args.suite] if args.suite else list(RCAEVAL_SUITES)
        suite_scenarios: dict[str, list] = {}
        for s in suites:
            ds = cfg.loader(suites=[s])
            suite_scenarios[s] = ds.load_fault_scenarios()
            print(f"  {s}: {len(suite_scenarios[s])} scenarios")
    else:
        scenarios = cfg.loader().load_fault_scenarios()
        print(f"  {len(scenarios)} scenarios | model={args.model}")

    levels = ["none", "light"] if args.context == "all" else [args.context]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ctx_tag = "all" if args.context == "all" else args.context
    run_name = (
        f"{cfg.name}_{ctx_tag}_{SELECTION_POLICY}_dk{DK_DOC_TAG}_{SUMMARY_MODE}_"
        f"nruns{args.n_runs}_t{int(round(args.temperature*100)):03d}_{timestamp}"
    )
    session_id = f"{run_name}_{uuid.uuid4().hex[:6]}"
    print(f"Langfuse session_id: {session_id}")

    client = make_client(args.model)
    per_level_runs: dict[str, list[dict]] = {lv: [] for lv in levels}

    for run_idx in range(args.run_idx, args.run_idx + args.n_runs):
        print(f"\n{'#' * 60}\n# Run {run_idx} (temperature={args.temperature})\n{'#' * 60}")
        for level in levels:
            if cfg.name == "rcaeval":
                # Per-suite execution — RCAEval domain phrase + context vary per suite.
                for s, scens in suite_scenarios.items():
                    domain_phrase = rcaeval_domain_for_suite(s)
                    domain_context = load_context(rcaeval_context_for_suite(s, level))
                    metrics = run_one_block(
                        dataset_name=cfg.name, scenarios=scens, level=level,
                        domain_phrase=domain_phrase,
                        domain_context=domain_context,
                        model=args.model, workers=args.workers,
                        run_idx=run_idx, n_runs=args.n_runs,
                        temperature=args.temperature,
                        run_name=run_name, session_id=session_id,
                        dataset_root=cfg.dataset_root, time_unit=cfg.time_unit,
                        agg_path=agg_path, client=client, suite=s,
                    )
                    per_level_runs[level].append(metrics)
            else:
                domain_context = load_context(
                    cfg.context_path if level != "none" else None
                )
                metrics = run_one_block(
                    dataset_name=cfg.name, scenarios=scenarios, level=level,
                    domain_phrase=cfg.domain_phrase,
                    domain_context=domain_context,
                    model=args.model, workers=args.workers,
                    run_idx=run_idx, n_runs=args.n_runs,
                    temperature=args.temperature,
                    run_name=run_name, session_id=session_id,
                    dataset_root=cfg.dataset_root, time_unit=cfg.time_unit,
                    agg_path=agg_path, client=client,
                )
                per_level_runs[level].append(metrics)

    write_stability_summary(
        dataset_name=cfg.name, save_dir=save_dir,
        per_level_runs=per_level_runs, n_runs=args.n_runs,
        temperature=args.temperature, model=args.model,
        session_id=session_id, run_name=run_name,
    )

    if _LANGFUSE is not None:
        try:
            _LANGFUSE.flush()
        except Exception:
            pass


if __name__ == "__main__":
    main()
