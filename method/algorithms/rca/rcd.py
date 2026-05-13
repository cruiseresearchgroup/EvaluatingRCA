"""Root Cause Discovery (RCD) adapter from the pyrca library."""

import pandas as pd

from pyrca.analyzers.rcd import RCD, RCDConfig

from method.algorithms.rca.base import RCAAdapter
from method.algorithms.rca.pyrca_utils import preprocess_pair, ranked_names
from method.datasets.base import FaultScenario


class RCDAdapter(RCAAdapter):
    """Root Cause Discovery (RCD) adapter.

    Parameters
    ----------
    k : int
        Top-k root causes to return internally.
    alpha_limit : float
        Significance level for conditional independence tests.
    patch : int
        Number of rows to sum into each patch before fitting.
    """

    requires_graph = False

    # PyRCA's RCD does an internal k-means with ``k=5`` over the concatenated
    # train+test patches, so each side needs at least this many patches.
    _MIN_PATCHES = 5

    def __init__(self, k: int = 10, alpha_limit: float = 0.5, patch: int = 100):
        self.k = k
        self.alpha_limit = alpha_limit
        self.patch = patch

    def _effective_patch(self, n_train: int, n_test: int) -> int:
        """Largest patch size that still yields ``_MIN_PATCHES`` samples on
        the smaller of the two windows. Falls back to 1 if a window is too
        short to hit the floor.
        """
        smaller = min(n_train, n_test)
        if smaller < self._MIN_PATCHES:
            return 1
        return max(1, min(self.patch, smaller // self._MIN_PATCHES))

    def predict(self, scenario: FaultScenario, graph: pd.DataFrame | None = None) -> list[str]:
        # Split at the known fault boundary so "normal" and "abnormal"
        # windows map onto PyRCA's find_root_causes(normal_df, abnormal_df) API.
        diag = scenario.diagnosis_time
        normal = scenario.data.loc[scenario.data.index < diag]
        abnormal = scenario.data.loc[scenario.data.index >= diag]

        patch = self._effective_patch(len(normal), len(abnormal))
        X_train, X_test = preprocess_pair(normal, abnormal, patch)

        model = RCD(config=RCDConfig(k=self.k, alpha_limit=self.alpha_limit))
        results = model.find_root_causes(X_train, X_test).to_list()
        return ranked_names(results)
