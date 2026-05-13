# Evaluating RCA

Implements the paper's RCA pipeline and the baselines it compares against:
statistical (BARO, RCD, ε-Diagnosis), graph-based (PageRank, RandomWalk, CIRCA
over per-scenario PC/FCI graphs), and an LLM reranker over a balanced
magnitude / onset / state-change candidate set.

## Layout

```
method/
├── algorithms/
│   ├── rca/             BARO, RCD, ε-Diagnosis, CIRCA, PageRank, RandomWalk,
│   │                    the candidate retriever, the LLM reranker
│   └── cd/              Causal-discovery learners used by the graph baselines
│                        (PC, FCI, per-scenario fitter)
├── datasets/            Per-dataset loaders (WADI, SWaT, HVAC, RCAEval RE1)
│                        + cached PC/FCI graphs + cached LLM ranking outputs
├── evaluation/metrics.py   top@k, Avg@k
├── prompts/             Light domain-knowledge contexts per dataset
└── runners/
    ├── _common.py                              shared helpers
    ├── _datasets.py                            per-dataset config registry
    ├── run_baseline.py                         statistical + graph baselines
    ├── run_llm_balanced.py                     LLM reranker (no-DK + with-DK)
    ├── compute_retrieval_recall_cumulative.py  retrieval table
    └── plot_retrieval_saturation.py            saturation prose values
```

## Install

```bash
pip install -r requirements.txt
```

LLM-reranker runs additionally need a `.env` at repo root with `GROQ_API_KEY`
and (optionally) `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` /
`LANGFUSE_HOST`. The baseline runner does not need any API keys.

## Run

```bash
# Baseline rows (Tables 5 & 6)
python method/runners/run_baseline.py --dataset wadi
python method/runners/run_baseline.py --dataset swat
python method/runners/run_baseline.py --dataset hvac
python method/runners/run_baseline.py --dataset rcaeval             # all three RE1 suites

# LLM reranker rows (Tables 5 & 6, both no-DK and with-DK conditions)
python method/runners/run_llm_balanced.py --dataset wadi
python method/runners/run_llm_balanced.py --dataset rcaeval --suite RE1-OB

# Retrieval table + saturation curves (Section 4)
python method/runners/compute_retrieval_recall_cumulative.py
python method/runners/plot_retrieval_saturation.py
```

Outputs land in `method/results/`. Pre-fitted PC/FCI graphs ship under
`method/datasets/<ds>/graph_cache_per_scenario/` so the graph baselines reuse
cached graphs instead of re-fitting from scratch (re-fitting takes hours,
dominated by FCI on RCAEval RE1-TT).
