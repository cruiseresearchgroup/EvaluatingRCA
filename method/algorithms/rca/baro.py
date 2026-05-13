"""Baro RCA adapter — RobustScaler anomaly scoring.

Splits at ``diagnosis_time`` into normal/anomalous windows, then ranks
columns by the max absolute robust-scaled z-score of the anomalous window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from baro.utility import drop_constant

from method.algorithms.rca.base import RCAAdapter
from method.datasets.base import FaultScenario

# RCAEval original baro — imported lazily so the adapter works without RCAEval on path
def _rcaeval_baro(raw_df, inject_time: int, dataset_name: str) -> list[str]:
    """Call the original RCAEval baro() function directly for exact parity."""
    import sys
    from pathlib import Path
    rcaeval_root = Path(__file__).resolve().parents[3] / "RCAEval"
    if str(rcaeval_root) not in sys.path:
        sys.path.insert(0, str(rcaeval_root))
    from RCAEval.e2e.baro import baro
    result = baro(raw_df, inject_time=inject_time, dataset=dataset_name)
    return result["ranks"]


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class BaroAdapter(RCAAdapter):
    """Baro RCA using RobustScaler scoring."""

    requires_graph = False

    def predict(
        self,
        scenario: FaultScenario,
        graph: pd.DataFrame | None = None,
    ) -> list[str]:
        # Fast path: if the scenario carries original RCAEval raw data, delegate
        # directly to the original baro() for exact parity.
        if scenario.metadata.get("raw_df") is not None:
            return _rcaeval_baro(
                scenario.metadata["raw_df"],
                scenario.metadata["inject_time"],
                scenario.metadata["dataset_name"],
            )

        data = scenario.data.ffill().fillna(0)

        sensor_data = drop_constant(data)
        if sensor_data.empty:
            return list(data.columns)

        diag_time = scenario.diagnosis_time
        normal = sensor_data.loc[sensor_data.index < diag_time]
        anomal = sensor_data.loc[sensor_data.index >= diag_time]

        if normal.empty or anomal.empty:
            return list(data.columns)

        ranks: list[tuple[str, float]] = []
        for col in sensor_data.columns:
            a = normal[col].to_numpy(dtype=float)
            b = anomal[col].to_numpy(dtype=float)
            if len(a) == 0 or len(b) == 0:
                continue
            scaler = RobustScaler().fit(a.reshape(-1, 1))
            zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
            ranks.append((col, float(np.max(np.abs(zscores)))))

        ranks.sort(key=lambda x: x[1], reverse=True)
        return [col for col, _ in ranks]
