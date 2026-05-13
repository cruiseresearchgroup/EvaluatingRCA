"""Per-scenario causal-graph fitting helper.

Contract:
  - Window the scenario data into (last `length*60*hz` of normal) +
    (first `length*60*hz` of fault), concatenated.
  - Preprocess: drop_constant + drop_near_constant.
  - Fit the causal graph per scenario on this windowed slice.
  - Cache by (scenario_id, cd_name, window_minutes).

Per-scenario fits (rather than once-per-dataset) keep each fit small and
make graph-based baselines comparable across scenarios with different
candidate-column sets after preprocessing.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import pandas as pd

# Sample rate (Hz) for each dataset's primary historian/metric stream.
DATASET_SAMPLE_RATE_HZ: dict[str, float] = {
    "swat":    1.0,        # 1 sample per second
    "wadi":    1.0,        # 1 sample per second
    "hvac":    1.0 / 60.0, # 1 sample per minute
    "rcaeval": 0.5,        # 1 sample per 2 seconds
}

# Default window size (minutes per side of diagnosis_time) per dataset.
# Picked so each side has enough samples for a reasonable PC/FGES fit.
# Default 10 min per side gives 600 samples at 1 Hz / 300 samples at 0.5 Hz.
# HVAC expands to 120 min because 1-min resolution would otherwise yield
# only ~10 samples per side.
DEFAULT_WINDOW_MINUTES: dict[str, int] = {
    "swat":    10,    # 1 Hz × 10 min = 600 samples per side, 1200 total
    "wadi":    10,    # 1 Hz × 10 min = 600 samples per side, 1200 total
    "hvac":    120,   # 1/60 Hz × 120 min = 120 samples per side, 240 total
    "rcaeval": 10,    # 0.5 Hz × 10 min = 300 samples per side, 600 total
}

MIN_SAMPLES_PER_SIDE = 30


# ---------------------------------------------------------------------------
# Preprocess
# ---------------------------------------------------------------------------

def drop_constant(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns where every value equals the first."""
    if df.empty:
        return df
    return df.loc[:, (df != df.iloc[0]).any()]


def drop_near_constant(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Drop columns whose modal value covers >= `threshold` fraction of rows."""
    if df.empty:
        return df
    n = len(df)
    keep: list[str] = []
    for col in df.columns:
        vc = df[col].value_counts(dropna=False)
        if len(vc) == 0:
            continue
        if vc.iloc[0] / n < threshold:
            keep.append(col)
    return df[keep]


def preprocess_for_cd(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess for causal-discovery input."""
    df = drop_constant(df)
    df = drop_near_constant(df)
    return df


# ---------------------------------------------------------------------------
# Per-scenario windowing
# ---------------------------------------------------------------------------

def window_around_inject(scenario, window_minutes: int,
                         sample_rate_hz: float) -> pd.DataFrame:
    """Slice: last `n_per_side` of normal + first `n_per_side` of fault."""
    diag = int(scenario.diagnosis_time)
    n_per_side = max(int(round(window_minutes * 60 * sample_rate_hz)),
                     MIN_SAMPLES_PER_SIDE)
    data = scenario.data.ffill().fillna(0.0)
    normal = data.iloc[:diag].tail(n_per_side)
    fault = data.iloc[diag:].head(n_per_side)
    return pd.concat([normal, fault], ignore_index=True)


# ---------------------------------------------------------------------------
# Per-scenario fit (with cache)
# ---------------------------------------------------------------------------

def _cache_path(cache_dir: Path, scenario_id: str, cd_name: str,
                window_minutes: int) -> Path:
    safe_sid = scenario_id.replace("/", "_").replace("|", "_")
    return Path(cache_dir) / f"perscen_{cd_name}_{safe_sid}_w{window_minutes}.pkl"


def fit_graph_per_scenario(
    adapter, scenario, *,
    dataset: str | None = None,
    window_minutes: int | None = None,
    sample_rate_hz: float | None = None,
    cache_dir: Path | None = None,
    cd_name: str = "graph",
    verbose: bool = False,
) -> pd.DataFrame:
    """Fit one causal graph on a per-scenario windowed slice.

    Parameters
    ----------
    adapter : CDAdapter
        Instance of PCAdapter / FGESAdapter / FCIAdapter / PCMCIAdapter.
    scenario : FaultScenario
        Scenario whose data will be sliced and used for fitting.
    dataset : str, optional
        Used to look up default `window_minutes` and `sample_rate_hz`.
    window_minutes : int, optional
        Override default per-side window length in minutes.
    sample_rate_hz : float, optional
        Override default historian sample rate.
    cache_dir : Path, optional
        Cache fitted graphs at `<cache_dir>/perscen_<cd_name>_<scenario_id>_w<W>.pkl`.
    cd_name : str
        Used in cache filename and verbose log.

    Returns
    -------
    pd.DataFrame
        Binary adjacency matrix (node-name index/columns).
        Empty DataFrame if window has too few columns after preprocess.
    """
    if window_minutes is None:
        window_minutes = DEFAULT_WINDOW_MINUTES.get(dataset, 10)
    if sample_rate_hz is None:
        sample_rate_hz = DATASET_SAMPLE_RATE_HZ.get(dataset, 1.0)

    if cache_dir is not None:
        cache_path = _cache_path(cache_dir, scenario.scenario_id, cd_name, window_minutes)
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)

    windowed = window_around_inject(scenario, window_minutes, sample_rate_hz)
    cleaned = preprocess_for_cd(windowed)

    if verbose:
        print(f"    [{cd_name}] {scenario.scenario_id}: "
              f"{windowed.shape[0]} rows × {windowed.shape[1]} cols → "
              f"{cleaned.shape[1]} cols after preprocess")

    if cleaned.shape[1] < 2 or cleaned.shape[0] < 10:
        return pd.DataFrame()

    # Drop columns perfectly collinear with an already-kept column to avoid
    # singular correlation matrices in fisherz-type tests (common when binary
    # actuators are exact complements, e.g. OA+RA damper position).
    cleaned = _drop_collinear(cleaned)
    if cleaned.shape[1] < 2:
        return pd.DataFrame()

    try:
        graph = adapter.fit(cleaned)
    except Exception as e:
        if verbose:
            print(f"    [{cd_name}] {scenario.scenario_id}: fit failed — {e}")
        return pd.DataFrame()

    if cache_dir is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(graph, f)

    return graph


def _drop_collinear(df: pd.DataFrame, abs_corr_threshold: float = 0.9999) -> pd.DataFrame:
    """Drop columns whose abs-correlation with an already-kept column exceeds threshold.

    Single-pass O(p^2). Threshold near 1.0 only kills *exact* collinearity
    (e.g., OA damper = 1 - RA damper) rather than merely high correlation.
    """
    import numpy as np
    if df.shape[1] < 2:
        return df
    corr = df.corr().abs().fillna(0.0)
    keep: list[str] = []
    for col in df.columns:
        if all(corr.loc[col, k] < abs_corr_threshold for k in keep):
            keep.append(col)
    return df[keep]


def fit_graphs_per_scenario(
    cd_factories, scenario, *,
    dataset: str | None = None,
    window_minutes: int | None = None,
    sample_rate_hz: float | None = None,
    cache_dir: Path | None = None,
    verbose: bool = False,
) -> dict[str, pd.DataFrame]:
    """Convenience wrapper: fit a dict of named graphs for one scenario.

    `cd_factories` is a list of (name, adapter_class, kwargs_dict) — same shape
    used in run_wadi.py's CD_GRAPHS list. Failed fits return an empty DataFrame
    rather than raising.
    """
    out: dict[str, pd.DataFrame] = {}
    for cd_name, cd_cls, cd_kwargs in cd_factories:
        try:
            adapter = cd_cls(**cd_kwargs)
            out[cd_name] = fit_graph_per_scenario(
                adapter, scenario,
                dataset=dataset,
                window_minutes=window_minutes,
                sample_rate_hz=sample_rate_hz,
                cache_dir=cache_dir,
                cd_name=cd_name,
                verbose=verbose,
            )
        except Exception as e:
            print(f"  [{cd_name}] {scenario.scenario_id}: fit failed — {e}")
            out[cd_name] = pd.DataFrame()
    return out
