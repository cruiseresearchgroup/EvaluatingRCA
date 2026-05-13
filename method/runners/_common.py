"""Shared helpers for the unified runners.

Used by both `run_baseline.py` and `run_llm_balanced.py`. Anything reused
across datasets and runs lives here so each runner stays a thin orchestrator
around per-dataset config from `_datasets.py`.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from method.algorithms.rca.candidate_selection import select_candidates_balanced
from method.algorithms.rca.llm_ranking import (
    build_hybrid_clean_user_prompt,
    build_system_prompt,
    log_run_summary,
)
from method.datasets.base import FaultScenario
from method.evaluation.metrics import avg_at_k, top_at_k

warnings.filterwarnings("ignore")

try:
    from langfuse import get_client
    _LANGFUSE = get_client() if os.environ.get("LANGFUSE_PUBLIC_KEY") else None
except Exception:
    _LANGFUSE = None


# ---------------------------------------------------------------------------
# Top-level paths and shared constants
# ---------------------------------------------------------------------------

BHNP_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_MODE = "hybrid_clean"          # paper-aligned; baked into cache paths
SELECTION_POLICY = "balanced_k15"      # paper-aligned candidate selection
DK_DOC_TAG = "light"                   # paper-aligned (Light context variant)
TOP_N = 15


# ---------------------------------------------------------------------------
# Metric aggregation and CSV merge
# ---------------------------------------------------------------------------

def compute_metrics(
    scenarios: list[FaultScenario],
    predictions: list[list[str]],
    label: str,
) -> dict:
    """Compute top@1/3/5 and avg@5 for one (algorithm, scenarios) pair."""
    y_true = [sc.ground_truth_causes for sc in scenarios]
    out: dict = {"algorithm": label, "n_attacks": len(scenarios)}
    for k in (1, 3, 5):
        out[f"top@{k}"] = round(float(np.mean(
            [top_at_k(t, p, k) for t, p in zip(y_true, predictions)]
        )), 4)
    out["avg@5"] = round(avg_at_k(y_true, predictions, 5), 4)
    return out


def merge_aggregate(path: Path, new_row: dict) -> None:
    """Upsert one row into a results CSV, keyed by the `algorithm` column."""
    label = new_row["algorithm"]
    if path.exists():
        df = pd.read_csv(path)
        df = df[df["algorithm"] != label]
    else:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True, sort=False)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# LLM client + retry wrapper
# ---------------------------------------------------------------------------

def make_client(model: str):
    """Return an OpenAI-compatible client. Groq for `openai/...` models, else OpenAI."""
    from openai import OpenAI
    if model.startswith("openai/"):
        return OpenAI(
            api_key=os.environ.get("GROQ_API_KEY", ""),
            base_url="https://api.groq.com/openai/v1",
        )
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


def llm_call_with_retry(client, model, messages, kw, max_retries: int = 8):
    """Call the LLM with 429-aware retry that respects the server's wait hint."""
    last_err = None
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model, messages=messages, **kw,
            )
        except Exception as e:
            last_err = e
            msg = str(e)
            if "429" not in msg and "rate_limit" not in msg.lower():
                raise
            m = re.search(r"try again in ([\d.]+)\s*(ms|s)", msg)
            if m:
                wait = float(m.group(1))
                if m.group(2) == "ms":
                    wait /= 1000.0
                wait += 0.5
            else:
                wait = min(2 ** attempt, 30.0)
            time.sleep(min(wait, 60.0))
    raise last_err if last_err else RuntimeError("max retries exceeded")


def parse_ranked_json(content: str, cand_names: list[str]) -> list[str]:
    """Extract the ranked list from a JSON LLM response, restricted to `cand_names`.

    Falls back to the heuristic candidate order if parsing fails or the LLM
    omits items. Also backfills any candidates the LLM dropped so downstream
    top@k evaluation always sees a full list.
    """
    try:
        s = content.strip()
        if s.startswith("```"):
            s = s.split("```", 2)[1]
            if s.startswith("json"):
                s = s[4:]
            s = s.rsplit("```", 1)[0]
        obj = json.loads(s)
        ranked = obj.get("ranked", [])
        cand_set = set(cand_names)
        out = [m for m in ranked if m in cand_set]
        for m in cand_names:
            if m not in out:
                out.append(m)
        return out
    except Exception:
        return list(cand_names)


def run_suffix(run_idx: int, temperature: float) -> str:
    """Path-friendly tag distinguishing a run from the paper-aligned default."""
    if run_idx == 0 and abs(temperature) < 1e-9:
        return ""
    return f"_run{run_idx}_t{int(round(temperature * 100)):03d}"


def load_context(context_path: Path | None) -> str:
    """Read a Light context file. Returns empty string for the no-DK level."""
    if context_path is None:
        return ""
    return context_path.read_text() if context_path.exists() else ""


# ---------------------------------------------------------------------------
# Balanced K/3 anomaly summary text builder
# ---------------------------------------------------------------------------

def build_balanced_summary(
    scenario: FaultScenario, time_unit: str,
) -> tuple[str, list[str]]:
    """Build the candidate-list + anomaly-summary text for the LLM prompt.

    Uses `select_candidates_balanced(scenario, k=TOP_N)` so the pool is the
    K/3 magnitude + K/3 onset + K/3 state-change set, deduped to ≤ K=15.
    Returns ``(summary_text, candidate_names)``.
    """
    candidates = select_candidates_balanced(scenario, k=TOP_N)
    cand_names = [m for m, _, _ in candidates]

    data = scenario.data.ffill().fillna(0)
    diag = int(scenario.diagnosis_time)
    normal = data.iloc[:diag]
    anomal = data.iloc[diag:]
    b_mean = normal.mean()
    b_std = normal.std()

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
    return "\n".join(lines), cand_names


# ---------------------------------------------------------------------------
# Per-scenario LLM prediction with caching + Langfuse trace
# ---------------------------------------------------------------------------

def predict_one_scenario(
    scenario: FaultScenario,
    *,
    dataset_name: str,
    time_unit: str,
    system_prompt: str,
    level: str,
    model: str,
    client,
    cache_dir: Path,
    run_name: str,
    session_id: str,
    run_idx: int = 0,
    n_runs: int = 1,
    temperature: float = 0.0,
) -> tuple[str, list[str], str]:
    """Run one LLM RCA call. Returns ``(scenario_id, ranked, source)`` where
    ``source`` is ``"cache"`` or ``"fresh"``.

    Caches by ``hash(scenario_id + summary_text)``. The cache directory itself
    encodes ``run_idx + temperature + dk_level`` so different runs do not
    collide; see :func:`llm_cache_dir`.
    """
    summary_text, cand_names = build_balanced_summary(scenario, time_unit)
    user_prompt = build_hybrid_clean_user_prompt(summary_text)

    key = hashlib.md5((scenario.scenario_id + summary_text).encode()).hexdigest()
    cache_file = cache_dir / f"rank_{key}.json"
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            ranked = cached if isinstance(cached, list) else cached.get("ranked", [])
            if ranked:
                return scenario.scenario_id, ranked, "cache"
        except Exception:
            pass

    is_reasoning = model.startswith(("gpt-5", "o1", "o3"))
    if is_reasoning:
        kw = dict(max_completion_tokens=8000, reasoning_effort="medium")
    else:
        kw = dict(temperature=temperature, max_tokens=4096)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    trace_meta = {
        "model": model, "top_n": TOP_N, "time_unit": time_unit,
        "summary_mode": SUMMARY_MODE,
        "selection_policy": SELECTION_POLICY,
        "dk_level": level,
        "dk_doc": DK_DOC_TAG if level != "none" else None,
        "dk_chars": len(system_prompt),
        "system_prompt": system_prompt,
        "ground_truth_eval_only": list(scenario.ground_truth_causes),
        "n_candidates": len(cand_names),
        "run_idx": run_idx, "n_runs": n_runs, "temperature": temperature,
        "run_name": run_name, "session_id": session_id,
    }
    trace_tags = [
        "rca-one-shot", f"dataset={dataset_name}", f"dk={level}", model,
        f"summary_mode={SUMMARY_MODE}",
        f"selection_policy={SELECTION_POLICY}",
        f"dk_doc={DK_DOC_TAG if level != 'none' else 'none'}",
        f"run_idx={run_idx}", f"n_runs={n_runs}",
        f"temperature={temperature}", f"run={run_name}",
        scenario.scenario_id,
    ]

    if _LANGFUSE is not None:
        with _LANGFUSE.start_as_current_span(
            name=f"rca_one_shot::{scenario.scenario_id}",
            input={"scenario_id": scenario.scenario_id,
                   "n_candidates": len(cand_names),
                   "summary_mode": SUMMARY_MODE,
                   "system_prompt": system_prompt},
        ):
            try:
                _LANGFUSE.update_current_trace(
                    name=f"rca_one_shot::{scenario.scenario_id}",
                    session_id=session_id,
                    metadata=trace_meta,
                    tags=trace_tags,
                )
            except Exception:
                pass
            with _LANGFUSE.start_as_current_generation(
                name="llm_one_shot_call",
                model=model,
                input=messages,
                model_parameters={k: str(v) for k, v in kw.items()},
            ) as gen:
                resp = llm_call_with_retry(client, model, messages, kw)
                content = resp.choices[0].message.content or ""
                gen.update(
                    output=content,
                    usage_details={
                        "input": resp.usage.prompt_tokens if resp.usage else 0,
                        "output": resp.usage.completion_tokens if resp.usage else 0,
                        "total": resp.usage.total_tokens if resp.usage else 0,
                    } if resp.usage else None,
                )
            ranked = parse_ranked_json(content, cand_names)
            gt = scenario.ground_truth_causes
            gt_pos = next((i + 1 for i, m in enumerate(ranked) if m in gt), -1)
            try:
                _LANGFUSE.update_current_trace(
                    output={"ranked_top10": ranked[:10], "gt_position": gt_pos},
                )
                for k, name in [(1, "hit@1"), (3, "hit@3"), (5, "hit@5")]:
                    _LANGFUSE.score_current_trace(
                        name=name,
                        value=int(any(g in ranked[:k] for g in gt)),
                    )
                _LANGFUSE.score_current_trace(
                    name="reciprocal_rank",
                    value=(1.0 / gt_pos) if gt_pos > 0 else 0.0,
                )
            except Exception:
                pass
    else:
        resp = client.chat.completions.create(model=model, messages=messages, **kw)
        content = resp.choices[0].message.content or ""
        ranked = parse_ranked_json(content, cand_names)

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        cache_file.write_text(json.dumps({
            "ranked": ranked, "scenario_id": scenario.scenario_id,
            "model": model, "user_prompt": user_prompt,
            "anomaly_summary": summary_text, "raw_response": content,
        }, indent=2))
    except Exception:
        pass

    return scenario.scenario_id, ranked, "fresh"


def llm_cache_dir(
    dataset_root: Path, model: str, level: str, run_suf: str,
) -> Path:
    """Build the per-(dataset, model, level, run) LLM cache directory path."""
    return dataset_root / (
        f"llm_rca_cache_{model.replace('/', '_')}_{level}_"
        f"{SUMMARY_MODE}_{SELECTION_POLICY}_dk{DK_DOC_TAG}{run_suf}"
    )


# Re-export Langfuse helpers so runners don't have to import llm_ranking directly.
__all__ = [
    "BHNP_ROOT", "SUMMARY_MODE", "SELECTION_POLICY", "DK_DOC_TAG", "TOP_N",
    "compute_metrics", "merge_aggregate",
    "make_client", "llm_call_with_retry",
    "parse_ranked_json", "run_suffix", "load_context",
    "build_balanced_summary", "predict_one_scenario", "llm_cache_dir",
    "log_run_summary", "build_system_prompt",
]
