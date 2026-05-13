"""Random-Walk RCA adapter.

Ranks variables by visit frequency during a random walk on the causal graph
(edges inverted so the walker travels effect → cause and surfaces upstream
sources rather than downstream effects).
"""

from __future__ import annotations

import numpy as np
import networkx as nx
import pandas as pd

from method.algorithms.rca.base import RCAAdapter
from method.datasets.base import FaultScenario


def _random_walk(
    adj: np.ndarray,
    node_names: list[str] | None = None,
    num_loop: int | None = None,
    seed: int = 42,
) -> list[tuple[str, float]]:
    """Visit-frequency random walk on a causal graph for root-cause ranking.

    Edges are added inverted (effect → cause) so the walker traverses causal
    arrows backwards from a downstream node toward its parents. Returns
    ``(node_name, visit_fraction)`` tuples sorted by visit fraction.
    """
    if node_names is None:
        node_names = [f"X{i}" for i in range(len(adj))]
    node_num = len(node_names)
    if num_loop is None:
        num_loop = node_num * 10

    G = nx.DiGraph()
    for name in node_names:
        G.add_node(name)
    for a in range(node_num):
        for b in range(node_num):
            if adj[a, b] != 0:
                G.add_edge(node_names[b], node_names[a])

    rng = np.random.RandomState(seed)
    current_node = node_names[rng.randint(0, node_num)]
    visits = {node: 0 for node in node_names}
    for _ in range(num_loop):
        neighbors = list(G.successors(current_node))
        if not neighbors:
            current_node = node_names[rng.randint(0, node_num)]
        else:
            current_node = neighbors[rng.randint(0, len(neighbors))]
        visits[current_node] += 1

    scores = [(node, count / num_loop) for node, count in visits.items()]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


class RandomWalkAdapter(RCAAdapter):
    """Random-walk-based RCA — requires a causal graph."""

    requires_graph = True

    def __init__(
        self,
        num_loop: int | None = None,
        random_seed: int = 42,
    ):
        self.num_loop = num_loop
        self.random_seed = random_seed

    def predict(
        self,
        scenario: FaultScenario,
        graph: pd.DataFrame | None = None,
    ) -> list[str]:
        if graph is None:
            raise ValueError("RandomWalkRCA requires a causal graph")

        sensor_cols = list(scenario.data.columns)
        graph_nodes = graph.columns.tolist()

        common = sorted(set(sensor_cols) & set(graph_nodes))
        if not common:
            return list(sensor_cols)

        idx = [graph_nodes.index(n) for n in common]
        sub_adj = graph.values[np.ix_(idx, idx)]

        ranks = _random_walk(
            adj=sub_adj,
            node_names=common,
            num_loop=self.num_loop,
            seed=self.random_seed,
        )
        return [n for n, _ in ranks]
