from abc import ABC, abstractmethod
import pandas as pd


class CDAdapter(ABC):
    """Abstract interface for causal discovery algorithm adapters."""

    @abstractmethod
    def fit(self, normal_data: pd.DataFrame) -> pd.DataFrame:
        """Learn a causal graph from normal-operation data.

        Parameters
        ----------
        normal_data : pd.DataFrame
            Wide-format DataFrame (time_s index × sensor columns, float values).

        Returns
        -------
        pd.DataFrame
            Binary adjacency matrix (node × node).  adj[i, j] = 1 means
            node i → node j.
        """
