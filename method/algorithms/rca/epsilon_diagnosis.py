"""Epsilon Diagnosis adapter from the pyrca library."""

import pandas as pd

from pyrca.analyzers.epsilon_diagnosis import EpsilonDiagnosis, EpsilonDiagnosisConfig

from method.algorithms.rca.base import RCAAdapter
from method.algorithms.rca.pyrca_utils import preprocess_pair, ranked_names
from method.datasets.base import FaultScenario


class EpsilonDiagnosisAdapter(RCAAdapter):
    """Epsilon Diagnosis adapter.

    Parameters
    ----------
    root_cause_top_k : int
        Number of top root causes to return.
    patch : int
        Number of rows to sum into each patch before fitting.
    """

    requires_graph = False

    def __init__(self, root_cause_top_k: int = 10, patch: int = 100):
        self.root_cause_top_k = root_cause_top_k
        self.patch = patch

    def predict(self, scenario: FaultScenario, graph: pd.DataFrame | None = None) -> list[str]:
        # Split at the known fault boundary.
        diag = scenario.diagnosis_time
        normal = scenario.data.loc[scenario.data.index < diag]
        abnormal = scenario.data.loc[scenario.data.index >= diag]
        X_train, X_test = preprocess_pair(normal, abnormal, self.patch)

        model = EpsilonDiagnosis(config=EpsilonDiagnosisConfig(root_cause_top_k=self.root_cause_top_k))
        model.train(X_train)
        results = model.find_root_causes(X_test).to_list()
        return ranked_names(results)
