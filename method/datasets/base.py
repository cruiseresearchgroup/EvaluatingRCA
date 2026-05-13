from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class FaultScenario:
    """Everything needed to evaluate one fault/attack event."""
    scenario_id: str
    data: pd.DataFrame              # wide-format: time index × sensor columns
    diagnosis_time: float            # seconds offset into `data` when RCA triggers
    ground_truth_causes: list[str]   # true root-cause variable names
    alarm_nodes: list[str]           # observed anomalous sensors
    description: str = ""
    raw_data: pd.DataFrame | None = None  # long-format (time_s, node, value, type) if available
    metadata: dict = field(default_factory=dict)  # dataset-specific extras (e.g. inject_time, dataset_name, raw_df)


class BenchmarkDataset(ABC):
    """Abstract interface that every dataset adapter must implement."""

    @abstractmethod
    def load_normal_data(self) -> pd.DataFrame:
        """Return wide-format DataFrame of normal operation.

        Columns are sensor/variable names, index is a numeric time axis
        (seconds).  Values are float.  Used as input to causal discovery.
        """

    @abstractmethod
    def load_fault_scenarios(self) -> list[FaultScenario]:
        """Return one FaultScenario per attack / fault event."""

    @abstractmethod
    def get_variable_names(self) -> list[str]:
        """Return all sensor/variable column names in the dataset."""

    def load_fault_scenario_datasets(self) -> list[pd.DataFrame]:
        """Return each fault scenario's data as a wide-format DataFrame for CD.

        Default implementation extracts .data from each FaultScenario.
        Datasets that need custom behaviour can override this method.
        """
        return [s.data for s in self.load_fault_scenarios()]


