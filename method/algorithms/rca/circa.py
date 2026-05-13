"""CIRCA RCA adapter.

CIRCA (Causal Inference-Based Root Cause Analysis) from KDD'22.
Paper: https://doi.org/10.1145/3534678.3539041
Source: https://github.com/NetManAIOps/CIRCA (installed as package)

Maps our FaultScenario → CIRCA's CaseData, optionally using a pre-built
causal graph from the CD pipeline (StaticGraphFactory) or no graph
(EmptyGraphFactory, falls back to NSigmaScorer).

Node convention: Node(entity=col_name, metric="value") for each sensor col.
SLI: the first (most anomalous) alarm node; falls back to first column.
"""

from collections import defaultdict

import networkx as nx
import pandas as pd

from circa.alg.ci import DAScorer, RHTScorer
from circa.alg.ci.anm import ANMRegressor
from circa.alg.common import Model, NSigmaScorer
from circa.graph.common import EmptyGraphFactory, StaticGraphFactory
from circa.model.case import CaseData
from circa.model.data_loader import MemoryDataLoader
from circa.model.graph import MemoryGraph, Node
from sklearn.linear_model import LinearRegression

from method.algorithms.rca.base import RCAAdapter
from method.datasets.base import FaultScenario


def _col_to_node(col: str) -> Node:
    return Node(entity=col, metric="value")


def _build_memory_graph(adj: pd.DataFrame, cols: list[str]) -> MemoryGraph:
    """Convert our adjacency DataFrame to CIRCA MemoryGraph."""
    col_set = set(cols)
    G = nx.DiGraph()
    nodes = [_col_to_node(c) for c in cols]
    G.add_nodes_from(nodes)
    for src in cols:
        for dst in cols:
            if src in adj.index and dst in adj.columns:
                if adj.loc[src, dst] != 0:
                    G.add_edge(_col_to_node(src), _col_to_node(dst))
    return MemoryGraph(G)


def _scenario_to_case_data(
    scenario: FaultScenario,
    sli_node: Node,
    lookup_window_min: int,
    detect_window_min: int,
) -> CaseData:
    """Convert FaultScenario wide-format data to CIRCA CaseData."""
    data = scenario.data  # index = float seconds from window start
    diag_time = scenario.diagnosis_time  # seconds

    # Build MemoryDataLoader dict: {entity: {metric: [(t, v), ...]}}
    loader_dict: dict = defaultdict(lambda: defaultdict(list))
    for col in data.columns:
        entity = col
        series = [(float(t), float(v)) for t, v in zip(data.index, data[col])]
        loader_dict[entity]["value"] = series

    loader = MemoryDataLoader(dict({k: dict(v) for k, v in loader_dict.items()}))

    # lookup_window: minutes of normal data before attack
    # detect_window: minutes of anomalous data after attack start
    total_secs = float(data.index[-1]) - diag_time
    detect_window = max(1, min(detect_window_min, int(total_secs / 60)))
    lookup_window = max(1, lookup_window_min)

    return CaseData(
        data_loader=loader,
        sli=sli_node,
        detect_time=diag_time,
        interval=__import__("datetime").timedelta(
            seconds=int(data.index[1] - data.index[0]) if len(data) > 1 else 60
        ),
        lookup_window=lookup_window,
        detect_window=detect_window,
        prune=True,
    )


class CIRCAAdapter(RCAAdapter):
    """CIRCA RCA adapter — uses RHTScorer (ANM) + DAScorer.

    Parameters
    ----------
    tau_max : int
        Maximum time lag for RHTScorer (0 = contemporaneous only).
    use_graph : bool
        If True and a graph is provided, use StaticGraphFactory.
        If False or no graph, use EmptyGraphFactory.
    lookup_window_min : int
        Minutes of normal history to use as training window.
    detect_window_min : int
        Minutes after diagnosis_time to use as anomaly window.
    n_sigma : float
        Fallback n-sigma threshold for NSigmaScorer when no graph.
    """

    requires_graph = False  # works with or without a graph

    def __init__(
        self,
        tau_max: int = 0,
        use_graph: bool = True,
        lookup_window_min: int = 30,
        detect_window_min: int = 10,
        n_sigma: float = 3.0,
    ):
        self.tau_max = tau_max
        self.use_graph = use_graph
        self.lookup_window_min = lookup_window_min
        self.detect_window_min = detect_window_min
        self.n_sigma = n_sigma

    def predict(
        self,
        scenario: FaultScenario,
        graph: pd.DataFrame | None = None,
    ) -> list[str]:
        cols = list(scenario.data.columns)

        # --- SLI: most anomalous alarm node, else first column ---
        sli_col = None
        for a in scenario.alarm_nodes:
            if a in cols:
                sli_col = a
                break
        if sli_col is None:
            sli_col = cols[0]
        sli_node = _col_to_node(sli_col)

        # --- Graph factory ---
        if self.use_graph and graph is not None:
            memory_graph = _build_memory_graph(graph, cols)
            graph_factory = StaticGraphFactory(memory_graph)
        else:
            # Build an empty graph (all nodes, no edges)
            G = nx.DiGraph()
            G.add_nodes_from([_col_to_node(c) for c in cols])
            graph_factory = StaticGraphFactory(MemoryGraph(G))

        # --- Scorers: RHT + DA ---
        scorers = [
            RHTScorer(
                tau_max=self.tau_max,
                regressor=ANMRegressor(regressor=LinearRegression()),
            ),
            DAScorer(),
        ]

        model = Model(graph_factory=graph_factory, scorers=scorers)

        # --- Build CaseData ---
        case_data = _scenario_to_case_data(
            scenario, sli_node, self.lookup_window_min, self.detect_window_min
        )

        # --- Run analysis ---
        try:
            current = scenario.diagnosis_time + self.detect_window_min * 60
            results = model.analyze(data=case_data, current=current)
        except Exception as e:
            print(f"  CIRCA error: {e} — falling back to unranked column order")
            return list(cols)

        # --- Convert output to ranked list of column names ---
        ranked = [node.entity for node, _ in results if node.entity in cols]
        # Append any columns not returned by CIRCA (keeps full ranking)
        ranked_set = set(ranked)
        ranked += [c for c in cols if c not in ranked_set]
        return ranked
