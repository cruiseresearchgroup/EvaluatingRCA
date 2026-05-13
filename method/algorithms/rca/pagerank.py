"""PageRank RCA adapter.

Ranks variables by their PageRank score in the causal graph.
Reimplemented for the benchmark's wide-format interface.
"""

import numpy as np
import networkx as nx
import pandas as pd
from sknetwork.ranking import PageRank

from method.algorithms.rca.base import RCAAdapter
from method.datasets.base import FaultScenario


class PageRankAdapter(RCAAdapter):
    """PageRank-based RCA — requires a causal graph."""

    requires_graph = True

    def predict(
        self,
        scenario: FaultScenario,
        graph: pd.DataFrame | None = None,
    ) -> list[str]:
        if graph is None:
            raise ValueError("PageRankRCA requires a causal graph")

        sensor_cols = list(scenario.data.columns)
        graph_nodes = graph.columns.tolist()

        # Intersect sensor columns with graph nodes
        common = [n for n in sensor_cols if n in graph_nodes]
        if not common:
            return list(sensor_cols)

        # Build sub-adjacency matrix for common nodes
        idx = [graph_nodes.index(n) for n in common]
        sub_adj = graph.values[np.ix_(idx, idx)]

        # Run PageRank on the transposed adjacency so random walks travel
        # toward upstream sources rather than downstream effects.
        pr = PageRank()
        scores = pr.fit_predict(sub_adj.T)

        ranked = sorted(zip(common, scores), key=lambda x: x[1], reverse=True)
        return [n for n, _ in ranked]
