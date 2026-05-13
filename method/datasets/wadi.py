"""WADI dataset adapter — rca_baselines-aligned preprocessing.

Key differences from the original combined-CSV approach:
  1. Loads from separate WADI_14days_new.csv (normal) and
     WADI_attackdataLABLE.csv (attack) files.
  2. Keeps only the 7 physical sensor types used by rca_baselines (lemma-rca):
     MV, LS, LT, FIT, AIT, MCV, P.
  3. Merges redundant 2A_*/2B_* dual sensors by summing them into 2_*.
  4. Uses a 30-min pre-attack window from the same day as baseline for
     each fault scenario.  The full 14-day normal file is available via
     load_normal_data() for causal discovery graph learning.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from method.datasets.base import BenchmarkDataset, FaultScenario

DATA_DIR = Path(__file__).resolve().parent / "wadi"

LABEL_COL  = "Attack LABLE (1:No Attack, -1:Attack)"
META_COLS  = {"Row", "Date", "Time", LABEL_COL}

# Physical sensor types to keep (rca_baselines convention)
KEEP_TYPES = {"MV", "LS", "LT", "FIT", "AIT", "MCV", "P"}

# Normal file starts at this datetime (row 1 = 2017-09-25 00:00:00)
NORMAL_START = pd.Timestamp("2017-09-25 00:00:00")
# Attack file starts at this datetime (row 1 = 2017-10-09 18:00:00)
ATTACK_START = pd.Timestamp("2017-10-09 18:00:00")

# Attack ground truth — targets use sensor names AFTER 2A/2B merge + filtering.
# Note: 2_PIC_003_SP (attack 13) is excluded by the 7-type filter; it will
# score 0 in evaluation (honest reporting of the limitation).
ATTACKS = [
    {"id": "1",   "start": "2017-10-09 19:25:00", "end": "2017-10-09 19:50:16",
     "targets": ["1_MV_001_STATUS"],
     "description": "MV-001 turned on → overflow on primary tank"},
    {"id": "2",   "start": "2017-10-10 10:24:10", "end": "2017-10-10 10:34:00",
     "targets": ["1_FIT_001_PV"],
     "description": "False reading on 1_FIT_001 → chemical dosing pump stays on"},
    {"id": "3",   "start": "2017-10-10 10:55:00", "end": "2017-10-10 11:24:00",
     "targets": ["2_LT_002_PV"],
     "description": "Stealthy: drain elevated reservoir via 2_LT_002"},
    {"id": "5",   "start": "2017-10-10 11:30:40", "end": "2017-10-10 11:44:50",
     "targets": ["2_MCV_101_CO", "2_MCV_201_CO", "2_MCV_301_CO",
                  "2_MCV_401_CO", "2_MCV_501_CO", "2_MCV_601_CO"],
     "description": "All consumer valves turned off"},
    {"id": "6",   "start": "2017-10-10 13:39:30", "end": "2017-10-10 13:50:40",
     "targets": ["2_MCV_101_CO", "2_MCV_201_CO"],
     "description": "MCV-101 and MCV-201 maliciously opened"},
    {"id": "7",   "start": "2017-10-10 14:48:17", "end": "2017-10-10 14:59:55",
     "targets": ["1_AIT_002_PV", "2_MV_003_STATUS"],
     "description": "Contaminated water to ER + open 2_MV_003"},
    {"id": "8",   "start": "2017-10-10 17:40:00", "end": "2017-10-10 17:49:40",
     "targets": ["2_MCV_007_CO"],
     "description": "MCV-007 opened → water leakage"},
    {"id": "9",   "start": "2017-10-11 10:55:00", "end": "2017-10-11 10:56:27",
     "targets": ["1_P_005_STATUS", "1_P_006_STATUS"],
     "description": "1_P_005 ON, 1_P_006 OFF → pipe burst"},
    {"id": "10",  "start": "2017-10-11 11:17:54", "end": "2017-10-11 11:31:20",
     "targets": ["1_MV_001_STATUS"],
     "description": "Damage 1_MV_001 and raw water pump → ER tank drains"},
    {"id": "11",  "start": "2017-10-11 11:36:31", "end": "2017-10-11 11:47:00",
     "targets": ["2_MCV_007_CO"],
     "description": "2MCV007 opened → booster never turns on"},
    {"id": "12",  "start": "2017-10-11 11:59:00", "end": "2017-10-11 12:05:00",
     "targets": ["2_MCV_007_CO"],
     "description": "2MCV007 gradually opened to 100% → water wasted"},
    {"id": "13",  "start": "2017-10-11 12:07:30", "end": "2017-10-11 12:10:52",
     "targets": ["2_PIC_003_SP"],          # PIC excluded by 7-type filter
     "description": "Reduce booster setpoint → intermittent consumer supply"},
    {"id": "14",  "start": "2017-10-11 12:16:00", "end": "2017-10-11 12:25:36",
     "targets": ["1_P_001_STATUS", "1_P_003_STATUS"],
     "description": "Stop chemical dosing pumps"},
    {"id": "15",  "start": "2017-10-11 15:26:30", "end": "2017-10-11 15:37:00",
     "targets": ["2_LT_002_PV"],
     "description": "Stealthy: inverse of attack 3 → overflow ER tank"},
]

# Baseline window length from clean normal day
BASELINE_MINUTES = 30
# Top-N anomalous sensors reported as alarm_nodes per attack
TOP_N_ALARMS = 40


# ---------------------------------------------------------------------------
# Helpers: loading and preprocessing
# ---------------------------------------------------------------------------

def _load_and_preprocess(path: Path, start_dt: pd.Timestamp) -> pd.DataFrame:
    """Load a WADI CSV, assign datetime, filter to 7 types, merge 2A/2B."""
    # Attack file has an extra numeric header row; detect and skip it
    peek = pd.read_csv(path, nrows=1, header=None)
    skip = 1 if str(peek.iloc[0, 0]).strip() == "0" else 0
    df = pd.read_csv(path, skiprows=skip, low_memory=False)

    # Strip whitespace / path prefixes from column names (e.g. \\2_AIT_001_PV)
    df.columns = [c.strip().split("\\")[-1] for c in df.columns]

    # Assign datetime from row index (1-second resolution, row 1 = start_dt)
    row_idx = pd.to_numeric(df["Row"], errors="coerce").fillna(1) - 1
    df["datetime"] = start_dt + pd.to_timedelta(row_idx.astype(int), unit="s")

    # Drop all-NaN columns
    df = df.dropna(axis=1, how="all")

    # Keep only physical sensor types
    keep_cols = {"Row", "Date", "Time", "datetime"}
    if LABEL_COL in df.columns:
        keep_cols.add(LABEL_COL)
    for c in df.columns:
        parts = c.split("_")
        if len(parts) >= 3 and parts[1] in KEEP_TYPES:
            keep_cols.add(c)
    df = df[[c for c in df.columns if c in keep_cols]]

    # Merge 2A_* and 2B_* dual sensors → 2_* (sum)
    a2_cols = sorted(c for c in df.columns if c.startswith("2A_"))
    b2_cols = sorted(c for c in df.columns if c.startswith("2B_"))
    for a, b in zip(a2_cols, b2_cols):
        merged_name = a.replace("2A_", "2_")
        df[merged_name] = pd.to_numeric(df[a], errors="coerce").fillna(0) \
                        + pd.to_numeric(df[b], errors="coerce").fillna(0)
        df = df.drop(columns=[a, b])

    # Fill remaining NaN with 0
    df = df.fillna(0)

    return df


def _sensor_cols(df: pd.DataFrame) -> list[str]:
    """Return sorted list of sensor column names (excludes meta + datetime)."""
    exclude = META_COLS | {"datetime"}
    return sorted(
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    )


def _detect_alarm_nodes(
    attack_df: pd.DataFrame,
    normal_df: pd.DataFrame,
    cols: list[str],
    attack_start: pd.Timestamp,
    attack_end: pd.Timestamp,
    baseline_minutes: int = BASELINE_MINUTES,
    top_n: int = TOP_N_ALARMS,
    z_threshold: float = 3.0,
) -> list[str]:
    """Return anomalous sensors ranked by earliest anomaly time (ascending).

    Baseline = pre-attack window on the same day (same file, minutes before
    attack_start).  Using the same-day baseline removes day-to-day drift in
    continuous quality sensors (AIT) that would otherwise dominate the ranking.

    All sensors whose z-score ever exceeds z_threshold during the attack are
    included (up to top_n cap).  Ranking by *first* anomaly time promotes
    upstream causes (valves/pumps that flip at attack t=0) over downstream
    sensors (quality/level sensors that drift later).
    """
    avail = [c for c in cols if c in attack_df.columns]

    # Pre-attack window from the same day (not Oct-5)
    pre_start = attack_start - pd.Timedelta(minutes=baseline_minutes)
    baseline = attack_df.loc[
        (attack_df["datetime"] >= pre_start) & (attack_df["datetime"] < attack_start),
        avail,
    ]
    attack = attack_df.loc[
        (attack_df["datetime"] >= attack_start) & (attack_df["datetime"] <= attack_end),
        avail,
    ]
    if baseline.empty or attack.empty:
        return []

    b_mean = baseline.mean()
    b_std  = baseline.std()

    # Sensors with zero baseline variance (constant value, e.g. MV/P/MCV at rest):
    # z-score is undefined, but ANY deviation = definite anomaly.
    # Detect them separately with absolute-change threshold.
    zero_std_cols = set(b_std[b_std == 0].index)
    b_std_safe = b_std.replace(0, np.nan)
    z_scores = ((attack - b_mean) / b_std_safe).abs()

    first_anomaly: dict[str, int] = {}

    # Zero-std sensors: first row where value differs from baseline constant
    for col in zero_std_cols:
        if col not in attack.columns:
            continue
        baseline_val = b_mean[col]
        changed = attack[col][(attack[col] - baseline_val).abs() > 1e-4]
        if not changed.empty:
            first_anomaly[col] = int(changed.index[0])

    # Actuator sensors (MV, MCV, P) with non-zero std: use absolute change threshold.
    # A valve going from 40%→100% is always significant regardless of baseline variance.
    ACTUATOR_TYPES = {"MV", "MCV", "P"}
    ACTUATOR_ABS_THRESHOLD = 5.0  # MCV_CO: 5% change; MV/P: 0.5 state change already caught above

    for col in z_scores.columns:
        if col in first_anomaly:
            continue
        parts = col.split("_")
        if len(parts) >= 2 and parts[1] in ACTUATOR_TYPES:
            abs_change = (attack[col] - b_mean[col]).abs()
            changed = abs_change[abs_change >= ACTUATOR_ABS_THRESHOLD]
            if not changed.empty:
                first_anomaly[col] = int(changed.index[0])

    # Variable sensors: first row where z-score exceeds threshold
    for col in z_scores.columns:
        if col in first_anomaly:
            continue
        col_z = z_scores[col].dropna()
        exceeds = col_z[col_z >= z_threshold]
        if not exceeds.empty:
            first_anomaly[col] = int(exceeds.index[0])

    if not first_anomaly:
        # Fallback: no sensor crosses threshold → use peak-z ranking
        peak_z = z_scores.max().dropna()
        return peak_z.sort_values(ascending=False).head(top_n).index.tolist()

    # Rank by first anomaly time (earliest = most likely root cause)
    ranked = sorted(first_anomaly.items(), key=lambda x: x[1])
    return [col for col, _ in ranked[:top_n]]


# ---------------------------------------------------------------------------
# WADIDataset
# ---------------------------------------------------------------------------

class WADIDataset(BenchmarkDataset):
    """Adapter for WADI using separate normal/attack files with rca_baselines
    preprocessing: 7-type sensor filter, 2A/2B merge, 30-min pre-attack baseline."""

    def __init__(
        self,
        data_dir: str | Path = DATA_DIR,
        resample_seconds: int = 60,
        baseline_minutes: int = BASELINE_MINUTES,
        top_n_alarms: int = TOP_N_ALARMS,
    ):
        self.data_dir        = Path(data_dir)
        self.resample_seconds = resample_seconds
        self.baseline_minutes = baseline_minutes
        self.top_n_alarms     = top_n_alarms

        self._normal_df: pd.DataFrame | None = None
        self._attack_df: pd.DataFrame | None = None
        self._sensor_cols: list[str] | None  = None

    # -- lazy loader --------------------------------------------------------

    def _ensure_loaded(self):
        if self._normal_df is not None:
            return

        normal_path = self.data_dir / "WADI_14days_new.csv"
        attack_path = self.data_dir / "WADI_attackdataLABLE.csv"

        print(f"Loading {normal_path.name} …")
        normal_df = _load_and_preprocess(normal_path, NORMAL_START)

        print(f"Loading {attack_path.name} …")
        attack_df = _load_and_preprocess(attack_path, ATTACK_START)

        # Sensor columns = union of both files (after filter + merge)
        n_cols = set(_sensor_cols(normal_df))
        a_cols = set(_sensor_cols(attack_df))
        cols   = sorted(n_cols | a_cols)

        self._normal_df   = normal_df
        self._attack_df   = attack_df
        self._sensor_cols = cols

        print(
            f"  Normal: {len(normal_df):,} rows | "
            f"Attack: {len(attack_df):,} rows | "
            f"{len(cols)} sensors (7-type filter + 2A/2B merge)"
        )

    # -- public interface ---------------------------------------------------

    def get_variable_names(self) -> list[str]:
        self._ensure_loaded()
        return list(self._sensor_cols)

    def load_normal_data(self) -> pd.DataFrame:
        """Return full 14-day normal data sub-sampled to resample_seconds.

        Uses only the clean normal file — no attack contamination.
        """
        self._ensure_loaded()
        step = max(1, self.resample_seconds)
        sampled = self._normal_df.iloc[::step]

        # Keep only sensor columns present in normal file
        avail = [c for c in self._sensor_cols if c in sampled.columns]
        out = sampled[avail].copy()
        out.index = np.arange(len(out), dtype=float) * step
        out.index.name = "time_s"
        out = out.ffill().fillna(0)

        print(
            f"  Normal data: {len(out):,} rows × {out.shape[1]} cols "
            f"@ {step}s resolution"
        )
        return out

    def load_fault_scenarios(self) -> list[FaultScenario]:
        """Build one FaultScenario per attack.

        Each scenario's data = [30-min pre-attack window (same day)] + [attack window].
        Using same-day pre-attack baseline removes day-to-day AIT sensor drift
        that would otherwise dominate the z-score anomaly ranking.
        """
        self._ensure_loaded()
        normal_df  = self._normal_df
        attack_df  = self._attack_df
        sensor_cols = self._sensor_cols

        scenarios: list[FaultScenario] = []
        for atk in ATTACKS:
            t0 = pd.Timestamp(atk["start"])
            t1 = pd.Timestamp(atk["end"])

            # ── Attack window ──────────────────────────────────────────────
            atk_win = attack_df.loc[
                (attack_df["datetime"] >= t0) & (attack_df["datetime"] <= t1)
            ].copy()
            if atk_win.empty:
                print(f"  Warning: no attack data for attack {atk['id']}")
                continue

            # ── Pre-attack baseline: same day, 30 min before attack start ──
            base_start = t0 - pd.Timedelta(minutes=self.baseline_minutes)
            base_end   = t0

            base_win = attack_df.loc[
                (attack_df["datetime"] >= base_start) &
                (attack_df["datetime"] <  base_end)
            ].copy()
            if base_win.empty:
                print(f"  Warning: no pre-attack baseline for attack {atk['id']}, "
                      f"falling back to attack window only")
                base_win = pd.DataFrame(columns=atk_win.columns)

            # ── Combine: baseline then attack ─────────────────────────────
            combined = pd.concat([base_win, atk_win], ignore_index=True)

            avail = [c for c in sensor_cols if c in combined.columns]
            data  = combined[avail].copy()
            data.index = np.arange(len(data), dtype=float)   # 1-second steps
            data.index.name = "time_s"
            data = data.ffill().fillna(0)

            # diagnosis_time = length of baseline in seconds
            diag_time = float(len(base_win))

            # ── Alarm nodes ────────────────────────────────────────────────
            alarms = _detect_alarm_nodes(
                attack_df, normal_df, avail, t0, t1,
                baseline_minutes=self.baseline_minutes,
                top_n=self.top_n_alarms,
            )

            # Filter ground-truth targets to those present after preprocessing
            valid_targets = [t for t in atk["targets"] if t in avail]
            if not valid_targets:
                print(f"  Warning: attack {atk['id']} target(s) "
                      f"{atk['targets']} not in filtered sensor set — "
                      f"keeping original for scoring")
                valid_targets = atk["targets"]

            scenarios.append(
                FaultScenario(
                    scenario_id=atk["id"],
                    data=data,
                    diagnosis_time=diag_time,
                    ground_truth_causes=valid_targets,
                    alarm_nodes=alarms,
                    description=atk["description"],
                )
            )

        print(f"  Built {len(scenarios)} fault scenarios")
        return scenarios

    def load_fault_scenario_datasets(self) -> list[pd.DataFrame]:
        """Return each fault scenario's data sub-sampled to resample_seconds."""
        scenarios = self.load_fault_scenarios()
        step = max(1, self.resample_seconds)
        result = []
        for s in scenarios:
            ds = s.data.iloc[::step].copy()
            ds.index = np.arange(len(ds), dtype=float) * step
            result.append(ds)
        return result
