"""HVAC RTU dataset adapter — LBNL FDD Data Sets (ORNL Experimental RTU).

Data layout:
  - One CSV per fault scenario (1440 rows, 1-minute resolution, full day).
  - Matching ERTU_<Season>.csv = multi-day fault-free baseline per season.
  - Occupied hours: 7:00 AM – 10:00 PM (rows where hour ∈ [7, 21]).

FaultScenario layout:
  - data: [baseline_window (last BASELINE_MINUTES occupied rows from seasonal
           baseline)] + [fault occupied rows], integer-minute index.
  - diagnosis_time: float = number of baseline rows (= BASELINE_MINUTES).
  - alarm_nodes: top-N most anomalous sensors (z-score vs baseline).
  - ground_truth_causes: from FAULT_TRUTH mapping.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from method.datasets.base import BenchmarkDataset, FaultScenario

DATA_DIR = Path(__file__).resolve().parent / "hvac" / "LBNL_FDD_Data_Sets_RTU" / "ORNL_RTU"

FAULT_TRUTH: dict[str, list[str]] = {
    "SA_temp_bias":    ["RTU_SA_TEMP"],
    "OA_damper_stuck": ["RTU_OA_DMPR_DM"],
    "Inc_Eco_SP":      ["RTU_OA_DMPR_DM"],
}

SEASONS = ["Fall_2020", "Spring_2021", "Summer_2021", "Winter_2022"]

BASELINE_MINUTES = 90   # occupied baseline rows prepended to each scenario
TOP_N_ALARMS     = 15   # top-N anomalous sensors reported as alarm_nodes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=["NAN", "NaN"])
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime")
    return df


def _occupied(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df.index.hour >= 7) & (df.index.hour < 22)].copy()


def _numeric_sensor_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c != "OCCU_MOD" and pd.api.types.is_numeric_dtype(df[c])]


def _detect_alarm_nodes(
    baseline: pd.DataFrame,
    fault: pd.DataFrame,
    z_threshold: float = 3.0,
    top_n: int = TOP_N_ALARMS,
) -> tuple[list[str], dict[str, int], dict[str, tuple[str, str]]]:
    """Return (alarm_nodes, offsets_minutes, states).

    offsets: minutes after fault-window start when sensor first exceeded threshold.
    states:  (before_str, after_str) averages for the prompt.
    """
    cols = _numeric_sensor_cols(fault)
    avail = [c for c in cols if c in baseline.columns]

    b_mean = baseline[avail].mean()
    b_std  = baseline[avail].std()

    first_anomaly: dict[str, int] = {}

    for col in avail:
        std_val = b_std.get(col, 0.0)
        if pd.isna(std_val):
            std_val = 0.0
        atk_col = fault[col].dropna()
        if atk_col.empty:
            continue

        if std_val == 0:
            bval = b_mean[col]
            if not np.isfinite(bval):
                continue
            changed = atk_col[(atk_col - bval).abs() > 1e-4]
            if not changed.empty:
                pos = fault.index.get_loc(changed.index[0])
                first_anomaly[col] = int(pos)
        else:
            z = ((atk_col - b_mean[col]) / std_val).abs()
            exceeds = z[z >= z_threshold]
            if not exceeds.empty:
                pos = fault.index.get_loc(exceeds.index[0])
                first_anomaly[col] = int(pos)

    ranked = sorted(first_anomaly.items(), key=lambda x: x[1])[:top_n]
    alarm_nodes = [col for col, _ in ranked]
    offsets = {col: t for col, t in ranked}

    states: dict[str, tuple[str, str]] = {}
    for col in alarm_nodes:
        before = b_mean.get(col, 0.0)
        after_vals = fault[col].dropna()
        after = float(after_vals.mean()) if not after_vals.empty else before
        before = float(before) if np.isfinite(before) else 0.0
        after  = float(after)  if np.isfinite(after)  else 0.0
        states[col] = (f"{before:.4f}", f"{after:.4f}")

    return alarm_nodes, offsets, states


def _parse_fault_type(stem: str) -> str | None:
    for ft in FAULT_TRUTH:
        if stem.startswith(ft):
            return ft
    return None


def _parse_season(stem: str) -> str | None:
    for s in SEASONS:
        if stem.endswith(s):
            return s
    return None


# ---------------------------------------------------------------------------
# HVACDataset
# ---------------------------------------------------------------------------

class HVACDataset(BenchmarkDataset):
    """LBNL FDD ORNL RTU dataset adapter.

    Normal data = union of all occupied seasonal baselines (ERTU_*.csv).
    Fault scenarios = one per fault CSV, with a BASELINE_MINUTES window
    from the seasonal baseline prepended.
    """

    def __init__(
        self,
        data_dir: str | Path = DATA_DIR,
        baseline_minutes: int = BASELINE_MINUTES,
        top_n_alarms: int = TOP_N_ALARMS,
    ):
        self.data_dir         = Path(data_dir)
        self.baseline_minutes = baseline_minutes
        self.top_n_alarms     = top_n_alarms

        self._baselines:      dict[str, pd.DataFrame] = {}  # season → occupied df
        self._sensor_cols:    list[str] | None = None
        self._normal_wide:    pd.DataFrame | None = None

    # -- lazy loaders -------------------------------------------------------

    def _load_baselines(self):
        if self._baselines:
            return
        for season in SEASONS:
            p = self.data_dir / f"ERTU_{season}.csv"
            if p.exists():
                self._baselines[season] = _occupied(_load_csv(p))

    def _build_sensor_cols(self):
        if self._sensor_cols is not None:
            return
        self._load_baselines()
        cols: set[str] = set()
        for df in self._baselines.values():
            cols.update(_numeric_sensor_cols(df))
        self._sensor_cols = sorted(cols)

    # -- public interface ---------------------------------------------------

    def get_variable_names(self) -> list[str]:
        self._build_sensor_cols()
        return list(self._sensor_cols)

    def load_normal_data(self) -> pd.DataFrame:
        """Return union of all occupied seasonal baselines as a wide DataFrame.

        Index is integer (0, 1, …) representing 1-minute steps.
        """
        if self._normal_wide is not None:
            return self._normal_wide

        self._load_baselines()
        self._build_sensor_cols()

        frames = []
        for season, df in sorted(self._baselines.items()):
            avail = [c for c in self._sensor_cols if c in df.columns]
            frames.append(df[avail])

        combined = pd.concat(frames, ignore_index=True)
        avail = [c for c in self._sensor_cols if c in combined.columns]
        out = combined[avail].copy()

        # Sanitise: replace inf with NaN then forward-fill
        out = out.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

        out.index = np.arange(len(out), dtype=float)
        out.index.name = "time_min"
        self._normal_wide = out
        return out

    def load_fault_scenarios(self) -> list[FaultScenario]:
        self._load_baselines()
        self._build_sensor_cols()

        scenarios: list[FaultScenario] = []
        for csv_path in sorted(self.data_dir.glob("*.csv")):
            stem = csv_path.stem
            ft     = _parse_fault_type(stem)
            season = _parse_season(stem)
            if ft is None or season is None:
                continue  # skip ERTU baseline files

            if season not in self._baselines:
                continue
            baseline_occ = self._baselines[season]

            fault_df  = _occupied(_load_csv(csv_path))
            if fault_df.empty:
                continue

            # Baseline window: last BASELINE_MINUTES rows of seasonal baseline
            base_win = baseline_occ.tail(self.baseline_minutes)

            # Detect alarm nodes and states (fault_df relative to base_win)
            alarm_nodes, offsets, states = _detect_alarm_nodes(
                base_win, fault_df,
                top_n=self.top_n_alarms,
            )

            # Combine: baseline rows first, then fault rows
            # Use only sensor columns present in both
            avail = [c for c in self._sensor_cols
                     if c in base_win.columns and c in fault_df.columns]

            combined_data = pd.concat(
                [base_win[avail], fault_df[avail]], ignore_index=True
            )
            combined_data = combined_data.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
            combined_data.index = np.arange(len(combined_data), dtype=float)
            combined_data.index.name = "time_min"

            diag_time = float(len(base_win))

            intensity = stem.replace(ft + "_", "").replace("_" + season, "")
            scenario_id = f"{ft}|{intensity}|{season}"

            scenarios.append(
                FaultScenario(
                    scenario_id=scenario_id,
                    data=combined_data,
                    diagnosis_time=diag_time,
                    ground_truth_causes=FAULT_TRUTH[ft],
                    alarm_nodes=alarm_nodes,
                    description=f"{ft} intensity={intensity} season={season}",
                )
            )

        print(f"  Built {len(scenarios)} HVAC fault scenarios")
        return scenarios
