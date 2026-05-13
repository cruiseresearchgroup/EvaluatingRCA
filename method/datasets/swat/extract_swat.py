#!/usr/bin/env python3
"""Selectively extract SWaT.zip for the time-series RCA benchmark.

The full SWaT release contains tens of GB of network packet captures
(``Network/*.csv``) used for network-level IDS work — they are NOT used
for the sensor-level RCA task. This script extracts only what the RCA
benchmark needs:

  1. ``Physical/`` — SCADA sensor + actuator time-series spreadsheets.
  2. ``List_of_attacks_Final.{xlsx,pdf}`` — ground-truth attack metadata
     (start/end time, attack point, intent, etc.).
  3. The dataset description PDF.

Optionally (``--include-network``), Network/*.csv files whose first row
falls inside any attack window from the attack list are also extracted —
useful only if you genuinely need packet-level data correlated with
attacks. Default is to skip Network/ entirely.

The script always runs in two passes:
  Pass 1 — list which entries would be extracted, totals in GB, and asks
           you to confirm before touching disk.
  Pass 2 — extract.

Usage
-----
    python extract_swat.py \\
        --zip /srv/scratch/$USER/SWaT.zip \\
        --out /srv/scratch/$USER/SWaT
    python extract_swat.py --zip ... --out ... --list-only
    python extract_swat.py --zip ... --out ... --include-network
"""
import argparse
import fnmatch
import io
import sys
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


# Files we always want regardless of folder layout differences across
# SWaT releases. Patterns match against the full path inside the zip.
KEEP_PATTERNS = [
    "*/A Dataset to Support Research*.pdf",
    "*/List_of_attacks_Final.pdf",
    "*/List_of_attacks_Final.xlsx",
    "*/Physical/*",
    # Some releases place the spreadsheets at the SCADA root rather than
    # under Physical/. Catch those too.
    "*Normal*.xlsx",
    "*Normal*.csv",
    "*Attack*.xlsx",
    "*Attack*.csv",
]

NETWORK_PATTERNS = [
    "*/Network/*.csv",
    "*/Network/*.log",
]


def _matches_any(name: str, patterns: List[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def _load_attack_windows(zf: zipfile.ZipFile) -> pd.DataFrame:
    """Read List_of_attacks_Final.xlsx from inside the zip and return a
    DataFrame with columns ``start`` (datetime) and ``end`` (datetime).

    SWaT releases differ slightly in column names; this is tolerant of
    common variants ("Start Time", "start_time", "Start", etc.).
    """
    matches = [n for n in zf.namelist() if n.endswith("List_of_attacks_Final.xlsx")]
    if not matches:
        raise FileNotFoundError("List_of_attacks_Final.xlsx not in zip")
    with zf.open(matches[0]) as f:
        df = pd.read_excel(io.BytesIO(f.read()))
    cols = {c.lower().strip(): c for c in df.columns}
    start_col = next((cols[k] for k in cols if "start" in k), None)
    end_col   = next((cols[k] for k in cols if "end"   in k), None)
    if start_col is None or end_col is None:
        raise ValueError(
            f"could not find Start/End columns; got: {list(df.columns)}"
        )
    df = df[[start_col, end_col]].dropna()
    df.columns = ["start", "end"]
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["end"]   = pd.to_datetime(df["end"], errors="coerce")
    df = df.dropna()
    return df


def _peek_first_timestamp(zf: zipfile.ZipFile, name: str) -> Optional[pd.Timestamp]:
    """Read the first data row of a CSV inside the zip and try to parse
    a timestamp from column 0 or column 1. Returns NaT on failure."""
    try:
        with zf.open(name) as f:
            header = f.readline().decode(errors="ignore")
            row    = f.readline().decode(errors="ignore")
        if not row:
            return None
        first_field = row.split(",", 1)[0].strip().strip('"')
        ts = pd.to_datetime(first_field, errors="coerce")
        if pd.isna(ts):
            # Try column 1 instead
            cols = row.split(",", 2)
            if len(cols) > 1:
                ts = pd.to_datetime(cols[1].strip().strip('"'), errors="coerce")
        return None if pd.isna(ts) else ts
    except Exception:
        return None


def _network_overlaps_any_attack(
    zf: zipfile.ZipFile,
    name: str,
    windows: pd.DataFrame,
    margin_minutes: int = 60,
) -> bool:
    """Heuristic: a Network CSV is considered relevant if its first
    timestamp falls within ``margin_minutes`` of any attack window.

    Without reading the entire file we can't know its end timestamp, so
    we extend the window symmetrically by ``margin_minutes`` on both
    sides. This keeps recall high; the cost is some false positives.
    """
    ts = _peek_first_timestamp(zf, name)
    if ts is None:
        return False
    pad = pd.Timedelta(minutes=margin_minutes)
    for _, row in windows.iterrows():
        if (row["start"] - pad) <= ts <= (row["end"] + pad):
            return True
    return False


def _classify(zf: zipfile.ZipFile, include_network: bool) -> Tuple[list, list]:
    """Return (keep_infos, skip_infos)."""
    keep = []   # type: List[zipfile.ZipInfo]
    skip = []   # type: List[zipfile.ZipInfo]

    network_filter = None
    if include_network:
        try:
            windows = _load_attack_windows(zf)
            print(f"[network filter] loaded {len(windows)} attack windows")
            network_filter = windows
        except Exception as e:
            print(f"[network filter] disabled — could not parse attack list: {e}")

    for info in zf.infolist():
        if info.is_dir():
            continue
        name = info.filename
        if _matches_any(name, KEEP_PATTERNS):
            keep.append(info)
        elif _matches_any(name, NETWORK_PATTERNS) and network_filter is not None:
            if _network_overlaps_any_attack(zf, name, network_filter):
                keep.append(info)
            else:
                skip.append(info)
        else:
            skip.append(info)

    return keep, skip


def _human_gb(b: int) -> str:
    return f"{b / 1e9:.2f} GB"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True, help="path to SWaT.zip")
    ap.add_argument("--out", required=True, help="extraction destination dir")
    ap.add_argument("--list-only", action="store_true",
                    help="show what would be extracted; do not write files")
    ap.add_argument("--include-network", action="store_true",
                    help="also extract Network/*.csv files whose first "
                         "timestamp falls within any attack window")
    ap.add_argument("--yes", action="store_true",
                    help="skip the confirmation prompt before extracting")
    args = ap.parse_args()

    zip_path = Path(args.zip)
    out_dir = Path(args.out)

    if not zip_path.is_file():
        print(f"error: zip not found: {zip_path}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        keep, skip = _classify(zf, include_network=args.include_network)

        keep_bytes = sum(i.file_size for i in keep)
        skip_bytes = sum(i.file_size for i in skip)
        print()
        print(f"keep:  {len(keep):4d} files   unpacked={_human_gb(keep_bytes)}")
        print(f"skip:  {len(skip):4d} files   unpacked={_human_gb(skip_bytes)}")
        print()

        if args.list_only:
            for i in keep:
                print(f"  KEEP {i.file_size / 1e6:8.1f} MB   {i.filename}")
            return

        if not args.yes:
            ans = input("Proceed with extraction? [y/N] ").strip().lower()
            if ans not in ("y", "yes"):
                print("aborted.")
                return

        for i, info in enumerate(keep, 1):
            print(f"  [{i}/{len(keep)}] {info.filename}  "
                  f"({info.file_size / 1e6:.1f} MB)")
            zf.extract(info, out_dir)

    # Post-extract: summarise the attack list so you know ground truth.
    candidates = list(out_dir.rglob("List_of_attacks_Final.xlsx"))
    if candidates:
        try:
            df = pd.read_excel(candidates[0])
            print()
            print("Attack list summary:")
            print(f"  rows: {len(df)}")
            print(f"  cols: {list(df.columns)}")
            print(df.head().to_string(index=False))
        except Exception as e:
            print(f"(could not read attack list: {e})")
    else:
        print("(no List_of_attacks_Final.xlsx found after extraction)")


if __name__ == "__main__":
    main()
