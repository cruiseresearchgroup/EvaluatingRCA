"""Per-dataset configuration registry.

Captures everything that differs across WADI / SWaT / HVAC / RCAEval in
one place so both `run_baseline.py` and `run_llm_balanced.py` can be one
script each, parameterised by `--dataset`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from method.datasets.base import BenchmarkDataset
from method.datasets.hvac import HVACDataset
from method.datasets.rcaeval_dataset import RCAEvalDataset, SUITES
from method.datasets.swat import SWaTDataset
from method.datasets.wadi import WADIDataset
from method.runners._common import BHNP_ROOT


DATASETS_ROOT = BHNP_ROOT / "method" / "datasets"
PROMPTS_ROOT = BHNP_ROOT / "method" / "prompts"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    loader: Callable[..., BenchmarkDataset]
    patch: int                            # PyRCA patch size for RCD / ε-Diagnosis
    time_unit: str                        # "s" or "min" (rendered in anomaly summary)
    service_level: bool                   # True only for RCAEval
    domain_phrase: str | Callable[[str], str]
                                          # Either a fixed string ("HVAC rooftop-unit
                                          # systems") OR, for RCAEval, a callable
                                          # ``suite -> domain phrase``.
    context_path: Path                    # Path to the Light context .md file
                                          # (returns "" when level == "none")
    graph_cache_dir: Path                 # Per-scenario CD graph cache
    dataset_root: Path                    # Root for LLM caches
    output_csv: str                       # "wadi_rca_results.csv" etc.


# RCAEval has a per-suite domain phrase
_RCAEVAL_SUITE_DOMAIN = {
    "RE1-OB": "an Online Boutique e-commerce microservice platform",
    "RE1-SS": "a Sock Shop e-commerce microservice platform",
    "RE1-TT": "a Train Ticket railway-ticketing microservice platform",
}


DATASETS: dict[str, DatasetConfig] = {
    "wadi": DatasetConfig(
        name="wadi",
        loader=WADIDataset,
        patch=60,
        time_unit="s",
        service_level=False,
        domain_phrase="water distribution ICS systems",
        context_path=PROMPTS_ROOT / "wadi" / "WADI_Context_Light.md",
        graph_cache_dir=DATASETS_ROOT / "wadi" / "graph_cache_per_scenario",
        dataset_root=DATASETS_ROOT / "wadi",
        output_csv="wadi_rca_results.csv",
    ),
    "swat": DatasetConfig(
        name="swat",
        loader=SWaTDataset,
        patch=60,
        time_unit="s",
        service_level=False,
        domain_phrase="water-treatment ICS systems",
        context_path=PROMPTS_ROOT / "swat" / "SWaT_Context_Light.md",
        graph_cache_dir=DATASETS_ROOT / "swat" / "graph_cache_per_scenario",
        dataset_root=DATASETS_ROOT / "swat",
        output_csv="swat_rca_results.csv",
    ),
    "hvac": DatasetConfig(
        name="hvac",
        loader=HVACDataset,
        patch=4,
        time_unit="min",
        service_level=False,
        domain_phrase="HVAC rooftop-unit systems",
        context_path=PROMPTS_ROOT / "hvac" / "HVAC_RTU_Context_Light.md",
        graph_cache_dir=DATASETS_ROOT / "hvac" / "graph_cache_per_scenario",
        dataset_root=DATASETS_ROOT / "hvac",
        output_csv="hvac_rca_results.csv",
    ),
    "rcaeval": DatasetConfig(
        name="rcaeval",
        # RCAEval needs the suites argument at construction time. The caller
        # passes a list of suites via the runner's --suite flag.
        loader=lambda suites=SUITES: RCAEvalDataset(suites=suites),
        patch=100,
        time_unit="s",
        service_level=True,
        # For RCAEval the domain phrase depends on the suite. The runner
        # resolves this per-suite via :func:`rcaeval_domain_for_suite`.
        domain_phrase=lambda suite: _RCAEVAL_SUITE_DOMAIN.get(
            suite, "a microservice platform"
        ),
        context_path=PROMPTS_ROOT / "rcaeval" / "OnlineBoutique_Context_Light.md",
        graph_cache_dir=DATASETS_ROOT / "rcaeval" / "graph_cache_per_scenario",
        dataset_root=DATASETS_ROOT / "rcaeval",
        output_csv="rcaeval_rca_results.csv",
    ),
}


# RCAEval-only helper: which context document to load for a given suite when
# constructing the with-DK system prompt.
RCAEVAL_SUITE_CONTEXT = {
    "RE1-OB": PROMPTS_ROOT / "rcaeval" / "OnlineBoutique_Context_Light.md",
    "RE1-SS": PROMPTS_ROOT / "rcaeval" / "SockShop_Context_Light.md",
    "RE1-TT": PROMPTS_ROOT / "rcaeval" / "TrainTicket_Context_Light.md",
}


def get(dataset_name: str) -> DatasetConfig:
    """Look up the per-dataset config or raise a friendly error."""
    cfg = DATASETS.get(dataset_name)
    if cfg is None:
        valid = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unknown dataset {dataset_name!r}. Valid: {valid}.")
    return cfg


def rcaeval_domain_for_suite(suite: str) -> str:
    """Resolve the with-DK domain phrase for one RCAEval suite."""
    return _RCAEVAL_SUITE_DOMAIN.get(suite, "a microservice platform")


def rcaeval_context_for_suite(suite: str, level: str) -> Path | None:
    """Resolve the with-DK Light context file for one RCAEval suite."""
    if level == "none":
        return None
    return RCAEVAL_SUITE_CONTEXT.get(suite)


__all__ = [
    "DatasetConfig", "DATASETS", "DATASETS_ROOT", "PROMPTS_ROOT",
    "RCAEVAL_SUITE_CONTEXT", "get",
    "rcaeval_domain_for_suite", "rcaeval_context_for_suite",
]
