from abc import ABC, abstractmethod
import pandas as pd

from method.datasets.base import FaultScenario


class RCAAdapter(ABC):
    """Abstract interface for root-cause analysis algorithm adapters."""

    requires_graph: bool = False
    is_supervised: bool = False

    @abstractmethod
    def predict(
        self,
        scenario: FaultScenario,
        graph: pd.DataFrame | None = None,
    ) -> list[str]:
        """Return a ranked list of root-cause candidates.

        Parameters
        ----------
        scenario : FaultScenario
            Contains the fault window data, diagnosis_time, alarm_nodes.
        graph : pd.DataFrame | None
            Binary adjacency matrix from CD stage.  Only provided when
            ``requires_graph`` is True.

        Returns
        -------
        list[str]
            Variable names ordered from most to least likely root cause.
        """


class SupervisedRCAAdapter(RCAAdapter):
    """Base class for supervised RCA adapters that need training."""

    is_supervised = True

    @abstractmethod
    def train(
        self,
        scenarios: list[FaultScenario],
        graph: pd.DataFrame | None = None,
    ) -> None:
        """Train on a set of labeled fault scenarios.

        Parameters
        ----------
        scenarios : list[FaultScenario]
            Training scenarios with known ground_truth_causes.
        graph : pd.DataFrame | None
            Causal graph (if requires_graph is True).
        """
