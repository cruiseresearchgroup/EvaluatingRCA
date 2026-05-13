"""FCI algorithm adapter — calls causallearn directly with correct directed conversion."""

import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.FCI import fci as causallearn_fci

from method.algorithms.cd.base import CDAdapter


def _causallearn_fci_to_directed_binary(graph_matrix: np.ndarray, node_names: list) -> pd.DataFrame:
    """Convert causallearn FCI -1/0/1/2 adjacency to directed binary adjacency.

    FCI convention (same as PC plus circle marks):
        graph[j,i]=1,  graph[i,j]=-1  →  i → j      (binary[i,j]=1)
        graph[i,j]=graph[j,i]=-1      →  i — j      (both)
        graph[i,j]=graph[j,i]=1       →  i <-> j    (both)
        graph[j,i]=1,  graph[i,j]=2   →  i o-> j    (treat as i→j)
        graph[i,j]=2,  graph[j,i]=1   →  i <-o j    (treat as i←j)
        graph[i,j]=graph[j,i]=2       →  i o-o j    (both)

    Logic mirrors RCAEval's page_rank_preprocess().
    """
    n = len(node_names)
    binary = np.zeros((n, n), dtype=float)
    for a in range(n):
        for b in range(n):
            va, vb = graph_matrix[a, b], graph_matrix[b, a]
            if va == 0 and vb == 0:
                pass
            elif va == -1 and vb == -1:          # undirected a -- b
                binary[a, b] = binary[b, a] = 1
            elif va == 1 and vb == -1:           # directed a -> b
                binary[a, b] = 1
            elif va == -1 and vb == 1:           # directed a <- b
                binary[b, a] = 1
            elif va == 1 and vb == 1:            # bidirected a <-> b
                binary[a, b] = binary[b, a] = 1
            elif va == 2 and vb == 1:            # a o-> b (treat as a->b)
                binary[a, b] = 1
            elif va == 1 and vb == 2:            # a <-o b (treat as a<-b)
                binary[b, a] = 1
            elif va == 2 and vb == 2:            # a o-o b (treat as bidirected)
                binary[a, b] = binary[b, a] = 1
    return pd.DataFrame(binary, index=node_names, columns=node_names)


class FCIAdapter(CDAdapter):
    """FCI algorithm — calls causallearn directly, preserving edge directionality."""

    def __init__(self, alpha: float = 0.05, indep_test: str = "fisherz"):
        # Defaults match RCAEval's fci_default (causallearn defaults).
        self.alpha = alpha
        self.indep_test = indep_test

    def fit(self, normal_data: pd.DataFrame) -> pd.DataFrame:
        print(f"  FCI: learning on {normal_data.shape[0]} rows × {normal_data.shape[1]} cols …")
        node_names = normal_data.columns.tolist()

        # Match RCAEval's fci_default: ffill then to_numpy, no np.abs() applied.
        data_clean = normal_data.ffill().fillna(0.0)
        np_data = data_clean.to_numpy().astype(float)

        G, _ = causallearn_fci(
            np_data,
            independence_test_method=self.indep_test,
            alpha=self.alpha,
            show_progress=False,
            verbose=False,
            node_names=node_names,
        )
        graph_matrix = G.graph  # shape (n, n), values in {-1, 0, 1, 2}

        adj = _causallearn_fci_to_directed_binary(graph_matrix, node_names)
        n_edges = int((adj.values != 0).sum())
        print(f"  FCI: found {n_edges} directed edges")
        return adj
