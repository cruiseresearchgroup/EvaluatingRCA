"""SWaT dataset adapter — mirrors the WADIDataset pattern.

SWaT (Secure Water Treatment, iTrust/SUTD) is the predecessor of WADI from
the same lab. The benchmark uses three xlsx files distributed with the
*A1 & A2 Dec 2015* release:

  - ``SWaT_Dataset_Normal_v1.xlsx`` — ~7 days of normal operation, 1 Hz,
    51 sensor channels + 1 ``Normal/Attack`` label column.
    (v1 is the corrected re-release of v0 with the first 30 minutes removed
    where the testbed was draining its raw-water tank for maintenance.)
  - ``SWaT_Dataset_Attack_v0.xlsx`` — ~4 days containing all 36 physical
    attacks back-to-back.
  - ``List_of_attacks_Final.xlsx`` — ground-truth attack list with start
    time, end time, attack point (the manipulated sensor / actuator), and
    attacker intent.

This class loads them, slices one fault scenario per physical attack, and
returns ``FaultScenario`` objects in the same shape as ``WADIDataset``.

Notes
-----
- The xlsx files are large (≈130 MB each) and slow to parse via openpyxl
  (~30–60 s per file). Parsed dataframes are cached to ``.feather`` next
  to the xlsx so subsequent loads are <1 s.
- Sensor column names in the xlsx have inconsistent leading whitespace
  (e.g. `' MV101'` vs `'P101'`); we strip them on load.
- Attack-point names in the attack list use hyphens (`MV-101`); data
  columns do not (`MV101`). We strip hyphens to match.
- Some rows in ``List_of_attacks_Final.xlsx`` are non-physical attacks or
  placeholders (NaN End Time, attack point ``"No Physical Impact Attack"``).
  These are skipped.
"""

import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from method.datasets.base import BenchmarkDataset, FaultScenario

DATA_DIR = Path(__file__).resolve().parent / "swat" / "SWaT.A1 & A2_Dec 2015"

NORMAL_XLSX      = "Physical/SWaT_Dataset_Normal_v1.xlsx"
ATTACK_XLSX      = "Physical/SWaT_Dataset_Attack_v0.xlsx"
ATTACK_LIST_XLSX = "List_of_attacks_Final.xlsx"


# Canonical SWaT attack catalogue. Baked-in from List_of_attacks_Final.xlsx
# (the file shipped with the A1 & A2 Dec 2015 release) and corrected for two
# documented quirks in the source spreadsheet:
#
#   - Attack 23 lists target "DIT-301"; the actual sensor is `DPIT301`.
#   - Attacks 37-41 list year "2015"; the testbed ran across the year
#     boundary and these attacks are dated "2016".
#
# Attack 4 targets `MV504`, which is mentioned in the published attack list
# but is NOT among the 51 channels recorded by the historian. The scenario
# is kept for completeness; it will score 0 in evaluation (honest reporting).
#
# Note that this list is also produced at runtime by ``_load_attack_list``
# from the xlsx; that helper is kept for re-baking should the upstream
# attack list change.
ATTACKS = [
    {"id": "1",  "start": "2015-12-28 10:29:14", "end": "2015-12-28 10:44:53",
     "targets": ["MV101"], "description": "Tank overflow"},
    {"id": "2",  "start": "2015-12-28 10:51:08", "end": "2015-12-28 10:58:30",
     "targets": ["P102"], "description": "Pipe bursts"},
    {"id": "3",  "start": "2015-12-28 11:22:00", "end": "2015-12-28 11:28:22",
     "targets": ["LIT101"], "description": "Tank underflow; damage P-101"},
    {"id": "4",  "start": "2015-12-28 11:47:39", "end": "2015-12-28 11:54:08",
     "targets": ["MV504"], "description": "Halt RO shut-down sequence; reduce life of RO  (MV504 not in historian)"},
    {"id": "6",  "start": "2015-12-28 12:00:55", "end": "2015-12-28 12:04:10",
     "targets": ["AIT202"], "description": "P-203 turns off; change in water quality"},
    {"id": "7",  "start": "2015-12-28 12:08:25", "end": "2015-12-28 12:15:33",
     "targets": ["LIT301"], "description": "Stop of inflow; tank underflow; damage P-301"},
    {"id": "8",  "start": "2015-12-28 13:10:10", "end": "2015-12-28 13:26:13",
     "targets": ["DPIT301"], "description": "Repeated backwash; normal operation stops"},
    {"id": "10", "start": "2015-12-28 14:16:20", "end": "2015-12-28 14:19:00",
     "targets": ["FIT401"], "description": "UV shutdown; P-501 turns off"},
    {"id": "11", "start": "2015-12-28 14:19:00", "end": "2015-12-28 14:28:20",
     "targets": ["FIT401"], "description": "UV shutdown; P-501 turns off"},
    {"id": "13", "start": "2015-12-29 11:11:25", "end": "2015-12-29 11:15:17",
     "targets": ["MV304"], "description": "Halt of stage 3 (backwash)"},
    {"id": "14", "start": "2015-12-29 11:35:40", "end": "2015-12-29 11:42:50",
     "targets": ["MV303"], "description": "Halt of stage 3 (backwash)"},
    {"id": "16", "start": "2015-12-29 11:57:25", "end": "2015-12-29 12:02:00",
     "targets": ["LIT301"], "description": "Tank overflow"},
    {"id": "17", "start": "2015-12-29 14:38:12", "end": "2015-12-29 14:50:08",
     "targets": ["MV303"], "description": "Halt of stage 3 (backwash)"},
    {"id": "19", "start": "2015-12-29 18:10:43", "end": "2015-12-29 18:15:01",
     "targets": ["AIT504"], "description": "RO shut-down sequence starts after 30 min"},
    {"id": "20", "start": "2015-12-29 18:15:43", "end": "2015-12-29 18:22:17",
     "targets": ["AIT504"], "description": "RO shut-down sequence starts after 30 min"},
    {"id": "21", "start": "2015-12-29 18:30:00", "end": "2015-12-29 18:42:00",
     "targets": ["MV101", "LIT101"], "description": "Tank overflow"},
    {"id": "22", "start": "2015-12-29 22:55:18", "end": "2015-12-29 23:03:00",
     "targets": ["UV401", "AIT502", "P501"], "description": "Possible damage to RO"},
    # Attack 23: source xlsx has typo "DIT-301"; corrected to DPIT301.
    {"id": "23", "start": "2015-12-30 01:42:34", "end": "2015-12-30 01:54:10",
     "targets": ["P602", "DPIT301", "MV302"], "description": "System freeze"},
    {"id": "24", "start": "2015-12-30 09:51:08", "end": "2015-12-30 09:56:28",
     "targets": ["P203", "P205"], "description": "Change in water quality"},
    {"id": "25", "start": "2015-12-30 10:01:50", "end": "2015-12-30 10:12:01",
     "targets": ["LIT401", "P401"], "description": "Tank underflow"},
    {"id": "26", "start": "2015-12-30 17:04:56", "end": "2015-12-30 17:29:00",
     "targets": ["P101", "LIT301"], "description": "Tank 101 underflow; tank 301 overflow"},
    {"id": "27", "start": "2015-12-31 01:17:08", "end": "2015-12-31 01:45:18",
     "targets": ["P302", "LIT401"], "description": "Tank overflow"},
    {"id": "28", "start": "2015-12-31 01:45:19", "end": "2015-12-31 11:15:27",
     "targets": ["P302"], "description": "Stop inflow of tank T-401"},
    {"id": "29", "start": "2015-12-31 15:32:00", "end": "2015-12-31 15:34:00",
     "targets": ["P201", "P203", "P205"], "description": "Wastage of chemicals"},
    {"id": "30", "start": "2015-12-31 15:47:40", "end": "2015-12-31 16:07:10",
     "targets": ["LIT101", "P101", "MV201"], "description": "Tank 101 underflow; tank 301 overflow"},
    {"id": "31", "start": "2015-12-31 22:05:34", "end": "2015-12-31 22:11:40",
     "targets": ["LIT401"], "description": "Tank overflow"},
    {"id": "32", "start": "2016-01-01 10:36:00", "end": "2016-01-01 10:46:00",
     "targets": ["LIT301"], "description": "Tank underflow; damage P-302"},
    {"id": "33", "start": "2016-01-01 14:21:12", "end": "2016-01-01 14:28:35",
     "targets": ["LIT101"], "description": "Tank underflow; damage P-101"},
    {"id": "34", "start": "2016-01-01 17:12:40", "end": "2016-01-01 17:14:20",
     "targets": ["P101"], "description": "Stops outflow"},
    {"id": "35", "start": "2016-01-01 17:18:56", "end": "2016-01-01 17:26:56",
     "targets": ["P101", "P102"], "description": "Stops outflow"},
    {"id": "36", "start": "2016-01-01 22:16:01", "end": "2016-01-01 22:25:00",
     "targets": ["LIT101"], "description": "Tank overflow"},
    # Attacks 37-41: source xlsx has year "2015"; corrected to "2016".
    {"id": "37", "start": "2016-01-02 11:17:02", "end": "2016-01-02 11:24:50",
     "targets": ["P501", "FIT502"], "description": "Reduced output"},
    {"id": "38", "start": "2016-01-02 11:31:38", "end": "2016-01-02 11:36:18",
     "targets": ["AIT402", "AIT502"], "description": "Water goes to drain (overdosing)"},
    {"id": "39", "start": "2016-01-02 11:43:48", "end": "2016-01-02 11:50:28",
     "targets": ["FIT401", "AIT502"], "description": "UV shuts down; water goes to RO"},
    {"id": "40", "start": "2016-01-02 11:51:42", "end": "2016-01-02 11:56:38",
     "targets": ["FIT401"], "description": "UV shuts down; water goes to RO"},
    {"id": "41", "start": "2016-01-02 13:13:02", "end": "2016-01-02 13:40:56",
     "targets": ["LIT301"], "description": "Tank overflow"},
]

# Header row index inside the Attack/Normal xlsx ("Combined Data" sheet
# has an empty row 0 and the real header on row 1).
HEADER_ROW = 1

LABEL_COL = "Normal/Attack"

# Tunables (match WADI defaults so cross-dataset comparisons are apples-to-apples)
BASELINE_MINUTES = 30
TOP_N_ALARMS     = 40

# Actuator types in SWaT (binary / staged): pumps, motorised valves, UV lamps.
ACTUATOR_TYPES = ("MV", "P", "UV")
# Threshold for "actuator changed" detection — pumps/valves are 0/1/2 in SWaT.
ACTUATOR_ABS_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _xlsx_to_cached(xlsx_path: Path, header_row: int) -> pd.DataFrame:
    """Read an xlsx, cache it as a pickle alongside, return the dataframe.

    Pickle is used (not feather/parquet) so we don't introduce a pyarrow
    dependency. The cache file lives next to the xlsx with a ``.pkl`` extension.
    """
    cache = xlsx_path.with_suffix(".pkl")
    if cache.exists():
        df = pd.read_pickle(cache)
    else:
        print(f"  Parsing {xlsx_path.name} (xlsx; ~30-60s) …")
        df = pd.read_excel(xlsx_path, header=header_row, engine="openpyxl")
        df = _strip_cols(df)
        for c in df.columns:
            if c.lower().startswith("timestamp"):
                continue
            if c == LABEL_COL:
                continue
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.to_pickle(cache)
        print(f"  Cached to {cache.name}")
    return df


def _normalise_attack_point(raw: str) -> List[str]:
    """Map an ``Attack Point`` cell to one or more data column names.

    Examples
    --------
    'MV-101'        -> ['MV101']
    'LIT-101'       -> ['LIT101']
    'P-101, P-102'  -> ['P101', 'P102']
    'P-101 ; P-102' -> ['P101', 'P102']
    'MV101'         -> ['MV101']
    """
    if not isinstance(raw, str):
        return []
    parts = re.split(r"[,;/]+", raw)
    out: List[str] = []
    for p in parts:
        s = re.sub(r"\s+", "", p)         # strip whitespace
        s = s.replace("-", "").replace("_", "")
        s = s.upper()
        if not s or "NOPHYSICAL" in s or "NOIMPACT" in s:
            continue
        # Keep tokens that look like a SWaT sensor: 2-4 letters + digits
        if re.match(r"^[A-Z]{1,4}\d+$", s):
            out.append(s)
    return out


def _parse_end_time(start: pd.Timestamp, end_cell) -> Optional[pd.Timestamp]:
    """End Time is often a ``datetime.time`` (Excel time-only cell) or a
    string like ``"10:44:53"``. Combine with the date from ``start`` so
    we don't accidentally get today's date.
    """
    import datetime as _dt
    if end_cell is None or (isinstance(end_cell, float) and pd.isna(end_cell)):
        return None
    # Excel time-only cells come through as datetime.time
    if isinstance(end_cell, _dt.time):
        return pd.Timestamp.combine(start.date(), end_cell)
    if isinstance(end_cell, pd.Timestamp):
        # Has both date and time; trust it only if the date isn't 1900 / today
        return end_cell
    s = str(end_cell).strip()
    # Full datetime: requires a date separator in the string
    if any(c in s for c in ("-", "/")) and len(s) >= 10:
        try:
            ts = pd.to_datetime(s, errors="raise")
            if not pd.isna(ts):
                return ts
        except Exception:
            pass
    # Time-only: combine with start's date
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            t = pd.to_datetime(s, format=fmt, errors="raise").time()
            return pd.Timestamp.combine(start.date(), t)
        except Exception:
            continue
    return None


def _load_attack_list(path: Path) -> List[dict]:
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    out: List[dict] = []
    for _, row in df.iterrows():
        try:
            start = pd.to_datetime(row.get("Start Time"), errors="coerce")
        except Exception:
            start = pd.NaT
        if pd.isna(start):
            continue
        end = _parse_end_time(start, row.get("End Time"))
        if end is None or end <= start:
            continue
        targets = _normalise_attack_point(str(row.get("Attack Point", "")))
        if not targets:
            continue
        out.append({
            "id": str(row.get("Attack #", "")).strip(),
            "start": start,
            "end": end,
            "targets": targets,
            "description": str(row.get("Expected Impact or attacker intent", "")).strip(),
            "raw_attack_point": str(row.get("Attack Point", "")).strip(),
        })
    return out


def _sensor_cols(df: pd.DataFrame) -> List[str]:
    """All numeric sensor columns (drops Timestamp / label / non-numeric)."""
    bad = {"Timestamp", LABEL_COL, "datetime"}
    return [
        c for c in df.columns
        if c not in bad and pd.api.types.is_numeric_dtype(df[c])
    ]


def _detect_alarm_nodes(
    attack_df: pd.DataFrame,
    cols: List[str],
    attack_start: pd.Timestamp,
    attack_end: pd.Timestamp,
    baseline_minutes: int = BASELINE_MINUTES,
    top_n: int = TOP_N_ALARMS,
    z_threshold: float = 3.0,
) -> List[str]:
    """Same-day pre-attack baseline + first-anomaly-time ranking.

    Mirrors ``wadi._detect_alarm_nodes`` so the two datasets produce
    comparable alarm-node sets.
    """
    pre_start = attack_start - pd.Timedelta(minutes=baseline_minutes)
    baseline = attack_df.loc[
        (attack_df["datetime"] >= pre_start) & (attack_df["datetime"] < attack_start),
        cols,
    ]
    attack = attack_df.loc[
        (attack_df["datetime"] >= attack_start) & (attack_df["datetime"] <= attack_end),
        cols,
    ]
    if baseline.empty or attack.empty:
        return []

    b_mean = baseline.mean()
    b_std = baseline.std()

    zero_std_cols = set(b_std[b_std == 0].index)
    b_std_safe = b_std.replace(0, np.nan)
    z_scores = ((attack - b_mean) / b_std_safe).abs()

    first_anomaly: dict = {}

    # Constant-baseline channels: any change is an anomaly
    for col in zero_std_cols:
        if col not in attack.columns:
            continue
        baseline_val = b_mean[col]
        changed = attack[col][(attack[col] - baseline_val).abs() > 1e-4]
        if not changed.empty:
            first_anomaly[col] = int(changed.index[0])

    # Actuators with non-zero std: absolute-change threshold (state flip)
    for col in z_scores.columns:
        if col in first_anomaly:
            continue
        prefix = re.match(r"^([A-Z]+)", col)
        if prefix and prefix.group(1) in ACTUATOR_TYPES:
            abs_change = (attack[col] - b_mean[col]).abs()
            changed = abs_change[abs_change >= ACTUATOR_ABS_THRESHOLD]
            if not changed.empty:
                first_anomaly[col] = int(changed.index[0])

    # Continuous variables: z-score crossing
    for col in z_scores.columns:
        if col in first_anomaly:
            continue
        col_z = z_scores[col].dropna()
        exceeds = col_z[col_z >= z_threshold]
        if not exceeds.empty:
            first_anomaly[col] = int(exceeds.index[0])

    if not first_anomaly:
        peak = z_scores.max().dropna()
        return peak.sort_values(ascending=False).head(top_n).index.tolist()

    ranked = sorted(first_anomaly.items(), key=lambda x: x[1])
    return [c for c, _ in ranked[:top_n]]


# ---------------------------------------------------------------------------
# SWaTDataset
# ---------------------------------------------------------------------------

class SWaTDataset(BenchmarkDataset):
    """Adapter for SWaT A1 & A2 Dec 2015 release."""

    def __init__(
        self,
        data_dir=DATA_DIR,
        resample_seconds: int = 60,
        baseline_minutes: int = BASELINE_MINUTES,
        top_n_alarms: int = TOP_N_ALARMS,
    ):
        self.data_dir         = Path(data_dir)
        self.resample_seconds = resample_seconds
        self.baseline_minutes = baseline_minutes
        self.top_n_alarms     = top_n_alarms

        self._normal_df = None      # type: Optional[pd.DataFrame]
        self._attack_df = None      # type: Optional[pd.DataFrame]
        self._sensor_cols = None    # type: Optional[List[str]]
        self._attacks = None        # type: Optional[List[dict]]

    # -- lazy loader --------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._normal_df is not None:
            return

        normal_path = self.data_dir / NORMAL_XLSX
        attack_path = self.data_dir / ATTACK_XLSX

        for p in (normal_path, attack_path):
            if not p.exists():
                raise FileNotFoundError(p)

        print(f"Loading SWaT from {self.data_dir} …")
        normal_df = _xlsx_to_cached(normal_path, HEADER_ROW)
        attack_df = _xlsx_to_cached(attack_path, HEADER_ROW)

        # Find timestamp column (case/whitespace-insensitive)
        for df, name in [(normal_df, "normal"), (attack_df, "attack")]:
            ts_col = next((c for c in df.columns if c.lower() == "timestamp"), None)
            if ts_col is None:
                raise RuntimeError(f"Timestamp column not found in {name} file")
            df["datetime"] = pd.to_datetime(df[ts_col], errors="coerce")

        # Sensor columns are the intersection of both files
        cols = sorted(set(_sensor_cols(normal_df)) & set(_sensor_cols(attack_df)))

        # Use the canonical hardcoded attack catalogue (with quirk fixes
        # already applied). The xlsx parser (`_load_attack_list`) is kept
        # in this module for re-baking should the upstream file change.
        attacks = [
            {**a,
             "start": pd.Timestamp(a["start"]),
             "end":   pd.Timestamp(a["end"]),
             "raw_attack_point": ", ".join(a["targets"])}
            for a in ATTACKS
        ]

        self._normal_df = normal_df
        self._attack_df = attack_df
        self._sensor_cols = cols
        self._attacks = attacks

        print(
            f"  Normal: {len(normal_df):,} rows | "
            f"Attack: {len(attack_df):,} rows | "
            f"{len(cols)} sensors | "
            f"{len(attacks)} attacks"
        )

    # -- public interface ---------------------------------------------------

    def get_variable_names(self) -> List[str]:
        self._ensure_loaded()
        return list(self._sensor_cols)

    def load_normal_data(self) -> pd.DataFrame:
        self._ensure_loaded()
        step = max(1, self.resample_seconds)
        sampled = self._normal_df.iloc[::step]
        out = sampled[self._sensor_cols].copy()
        out.index = np.arange(len(out), dtype=float) * step
        out.index.name = "time_s"
        out = out.ffill().fillna(0)
        print(
            f"  Normal data: {len(out):,} rows × {out.shape[1]} cols "
            f"@ {step}s resolution"
        )
        return out

    def load_fault_scenarios(self) -> List[FaultScenario]:
        self._ensure_loaded()
        attack_df = self._attack_df
        cols = self._sensor_cols

        scenarios: List[FaultScenario] = []
        for atk in self._attacks:
            t0, t1 = atk["start"], atk["end"]

            atk_win = attack_df.loc[
                (attack_df["datetime"] >= t0) & (attack_df["datetime"] <= t1)
            ].copy()
            if atk_win.empty:
                print(f"  Warning: no attack data for attack {atk['id']} (t0={t0})")
                continue

            base_start = t0 - pd.Timedelta(minutes=self.baseline_minutes)
            base_win = attack_df.loc[
                (attack_df["datetime"] >= base_start) &
                (attack_df["datetime"] <  t0)
            ].copy()

            combined = pd.concat([base_win, atk_win], ignore_index=True)
            data = combined[cols].copy()
            data.index = np.arange(len(data), dtype=float)
            data.index.name = "time_s"
            data = data.ffill().fillna(0)

            diag_time = float(len(base_win))

            alarms = _detect_alarm_nodes(
                attack_df, cols, t0, t1,
                baseline_minutes=self.baseline_minutes,
                top_n=self.top_n_alarms,
            )

            valid_targets = [t for t in atk["targets"] if t in cols]
            if not valid_targets:
                print(
                    f"  Warning: attack {atk['id']} target(s) {atk['targets']} "
                    f"not in sensor set — keeping originals for scoring"
                )
                valid_targets = atk["targets"]

            scenarios.append(FaultScenario(
                scenario_id=atk["id"],
                data=data,
                diagnosis_time=diag_time,
                ground_truth_causes=valid_targets,
                alarm_nodes=alarms,
                description=atk["description"],
                metadata={
                    "raw_attack_point": atk["raw_attack_point"],
                    "attack_start": str(t0),
                    "attack_end":   str(t1),
                },
            ))

        print(f"  Built {len(scenarios)} fault scenarios")
        return scenarios

    def load_fault_scenario_datasets(self) -> List[pd.DataFrame]:
        scenarios = self.load_fault_scenarios()
        step = max(1, self.resample_seconds)
        out = []
        for s in scenarios:
            ds = s.data.iloc[::step].copy()
            ds.index = np.arange(len(ds), dtype=float) * step
            out.append(ds)
        return out
