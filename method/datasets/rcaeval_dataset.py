"""RCAEval dataset adapter — RE1 suite (Online Boutique, Sock Shop, Train Ticket).

Data layout:
  benchmark/datasets/rcaeval/{suite}/{service}_{fault}/{instance}/
    data.csv        — 1-second time series, columns: time + {service}_{metric}
    inject_time.txt — Unix timestamp of fault injection

FaultScenario layout:
  - data: [last BASELINE_SECONDS of pre-inject window] + [POST_SECONDS after inject],
    integer-second index.
  - diagnosis_time: float = number of baseline rows.
  - alarm_nodes: top-N metrics ranked by earliest z>=3 anomaly vs baseline.
  - ground_truth_causes: ["{service}_{fault}"] from directory name.

Preprocessing matches original RCAEval main.py (--length 20 default):
  - Window: 600 rows each side  (20 min * 60s / 2s sampling = 600)
  - Drop lat-50 columns, rename _latency-90 -> _latency
  - Drop constant columns, convert memory columns from bytes to MB
"""

import numpy as np
import pandas as pd
from pathlib import Path

from method.datasets.base import BenchmarkDataset, FaultScenario

DATA_DIR = Path(__file__).resolve().parent / "rcaeval"

SUITES = ["RE1-OB", "RE1-SS", "RE1-TT"]

# Maps suite prefix → RCAEval dataset name used by original main.py
SUITE_DATASET_NAME = {
    "RE1-OB": "online-boutique",
    "RE2-OB": "online-boutique",
    "RE1-SS": "sock-shop-2",
    "RE2-SS": "sock-shop-2",
    "RE1-TT": "train-ticket",
    "RE2-TT": "train-ticket",
}

BASELINE_SECONDS = 600    # matches RCAEval default --length 20 (20*60//2 = 600 rows)
POST_SECONDS     = 600    # same
TOP_N_ALARMS     = 20     # top-N anomalous metrics as alarm_nodes


def _preprocess_rcaeval(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate the preprocessing used by the ASE'24 RCAEval main-ase.py.

    We intentionally do NOT strip ``_latency-50`` or rename
    ``_latency-90`` -> ``_latency`` — those transformations are applied
    by the newer ``main.py`` and cause Baro to score 0.57 on Sock Shop
    (vs. 0.95 published). Keeping raw per-quantile latency columns
    reproduces the paper's published numbers exactly.

    Steps (in order):
      1. Drop the 'time' column if present
      2. Drop constant columns  (preprocess() -> drop_constant)
      3. Convert memory columns bytes -> MB  (preprocess() -> convert_mem_mb)
    """
    df = df.copy()

    # 1. drop time column
    for tcol in ("time", "Time", "time.1"):
        if tcol in df.columns:
            df = df.drop(columns=[tcol])

    # 2. drop constant columns
    df = df.loc[:, (df != df.iloc[0]).any()]

    # 3. convert memory bytes -> MB
    for col in df.columns:
        if col.endswith("_mem"):
            df[col] = df[col] / 1e6

    return df


def _detect_alarm_nodes(
    baseline: pd.DataFrame,
    fault: pd.DataFrame,
    z_threshold: float = 3.0,
    top_n: int = TOP_N_ALARMS,
) -> tuple[list[str], dict[str, int]]:
    """Return (alarm_nodes, offsets) ranked by first anomaly time."""
    avail = [c for c in fault.columns
             if c in baseline.columns and pd.api.types.is_numeric_dtype(fault[c])]

    b_mean = baseline[avail].mean()
    b_std  = baseline[avail].std()

    first_anomaly: dict[str, int] = {}
    for col in avail:
        std_val = float(b_std.get(col, 0.0))
        if pd.isna(std_val) or std_val == 0:
            bval = float(b_mean[col]) if np.isfinite(b_mean.get(col, np.nan)) else None
            if bval is None:
                continue
            changed = fault[col].dropna()
            changed = changed[(changed - bval).abs() > 1e-4]
            if not changed.empty:
                first_anomaly[col] = int(fault.index.get_loc(changed.index[0]))
        else:
            z = ((fault[col].dropna() - b_mean[col]) / std_val).abs()
            exceeds = z[z >= z_threshold]
            if not exceeds.empty:
                first_anomaly[col] = int(fault.index.get_loc(exceeds.index[0]))

    ranked = sorted(first_anomaly.items(), key=lambda x: x[1])[:top_n]
    alarm_nodes = [col for col, _ in ranked]
    offsets = {col: t for col, t in ranked}
    return alarm_nodes, offsets


class RCAEvalDataset(BenchmarkDataset):
    """RE1 suite adapter (Online Boutique, Sock Shop, Train Ticket).

    Normal data = concatenation of all pre-inject baseline windows.
    """

    def __init__(
        self,
        data_dir: str | Path = DATA_DIR,
        suites: list[str] | None = None,
        baseline_seconds: int = BASELINE_SECONDS,
        post_seconds: int = POST_SECONDS,
        top_n_alarms: int = TOP_N_ALARMS,
    ):
        self.data_dir        = Path(data_dir)
        self.suites          = suites or SUITES
        self.baseline_seconds = baseline_seconds
        self.post_seconds    = post_seconds
        self.top_n_alarms    = top_n_alarms
        self._scenarios: list[FaultScenario] | None = None

    def _iter_cases(self):
        """Yield (suite, service, fault, instance, data_csv, inject_time) tuples."""
        for suite in self.suites:
            suite_dir = self.data_dir / suite
            if not suite_dir.exists():
                continue
            for fault_dir in sorted(suite_dir.iterdir()):
                if not fault_dir.is_dir():
                    continue
                # Parse "service_fault" — fault is the last underscore token
                parts = fault_dir.name.rsplit("_", 1)
                if len(parts) != 2:
                    continue
                service, fault = parts
                for inst_dir in sorted(fault_dir.iterdir()):
                    if not inst_dir.is_dir():
                        continue
                    # Prefer the aggregated CSV. simple_data.csv is used by
                    # ASE'24 main-ase.py for RE1-SS/RE1-TT. simple_metrics.csv
                    # is the RE2/RE3 equivalent (same {service}_{metric}
                    # schema). Fall back to raw data.csv (e.g., RE1-OB
                    # ships only the data.csv).
                    data_csv = None
                    for cand in ("simple_data.csv", "simple_metrics.csv", "data.csv"):
                        p = inst_dir / cand
                        if p.exists():
                            data_csv = p
                            break
                    inject_txt = inst_dir / "inject_time.txt"
                    if data_csv is None or not inject_txt.exists():
                        continue
                    inject_time = int(inject_txt.read_text().strip())
                    yield suite, service, fault, inst_dir.name, data_csv, inject_time

    def get_variable_names(self) -> list[str]:
        scenarios = self.load_fault_scenarios()
        cols: set[str] = set()
        for s in scenarios:
            cols.update(s.data.columns)
        return sorted(cols)

    def load_normal_data(self) -> pd.DataFrame:
        """Concatenate all pre-inject baseline windows as normal data."""
        frames = []
        for _, _, _, _, data_csv, inject_time in self._iter_cases():
            df = pd.read_csv(data_csv)
            df = df[df["time"] < inject_time].copy()
            if len(df) > self.baseline_seconds:
                df = df.tail(self.baseline_seconds)
            df = _preprocess_rcaeval(df)
            frames.append(df)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
        combined.index = np.arange(len(combined), dtype=float)
        combined.index.name = "time_s"
        return combined

    def load_fault_scenarios(self) -> list[FaultScenario]:
        if self._scenarios is not None:
            return self._scenarios

        scenarios = []
        for suite, service, fault, instance, data_csv, inject_time in self._iter_cases():
            df = pd.read_csv(data_csv)

            # Build the raw_df that the original RCAEval main-ase.py (ASE'24)
            # passes to algorithms: clip windows + inf/nan handling only.
            # We intentionally preserve raw per-quantile latency columns
            # (no _latency-50 strip, no _latency-90 rename) to match the
            # published ASE'24 numbers.
            raw_pre  = df[df["time"] < inject_time].tail(self.baseline_seconds).copy()
            raw_post = df[df["time"] >= inject_time].head(self.post_seconds).copy()
            raw_df = pd.concat([raw_pre, raw_post], ignore_index=True)
            raw_df = raw_df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

            pre  = df[df["time"] < inject_time].copy()
            post = df[df["time"] >= inject_time].copy()

            if pre.empty or post.empty:
                continue

            # Clip to configured windows before preprocessing
            base_win  = pre.tail(self.baseline_seconds)
            fault_win = post.head(self.post_seconds)

            # Apply RCAEval-matching preprocessing to each window independently,
            # then align columns (intersection) so shapes match.
            base_win  = _preprocess_rcaeval(base_win)
            fault_win = _preprocess_rcaeval(fault_win)

            common_cols = [c for c in base_win.columns if c in fault_win.columns]
            if not common_cols:
                continue
            base_win  = base_win[common_cols]
            fault_win = fault_win[common_cols]

            # Detect alarm nodes (preprocessing already dropped time/constant cols)
            alarm_nodes, _ = _detect_alarm_nodes(
                base_win.reset_index(drop=True),
                fault_win.reset_index(drop=True),
                top_n=self.top_n_alarms,
            )

            # Build combined data with integer index
            combined = pd.concat([base_win, fault_win], ignore_index=True)
            combined = combined.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
            combined.index = np.arange(len(combined), dtype=float)
            combined.index.name = "time_s"

            diag_time = float(len(base_win))
            scenario_id = f"{suite}|{service}_{fault}|{instance}"
            truth = f"{service}_{fault}"

            scenarios.append(FaultScenario(
                scenario_id=scenario_id,
                data=combined,
                diagnosis_time=diag_time,
                ground_truth_causes=[truth],
                alarm_nodes=alarm_nodes,
                description=f"{suite}: {service} {fault} fault (instance {instance})",
                metadata={
                    "raw_df":       raw_df,
                    "inject_time":  inject_time,
                    "dataset_name": SUITE_DATASET_NAME.get(suite, suite),
                    "suite":        suite,
                },
            ))

        print(f"  Built {len(scenarios)} RCAEval fault scenarios "
              f"({', '.join(self.suites)})")
        self._scenarios = scenarios
        return scenarios
