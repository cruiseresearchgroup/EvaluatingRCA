"""LLM Ranking RCA adapter — proposed method.

Uses hybrid candidate selection (z-score + early-onset + state-change) to build
an anomaly summary, then asks an LLM to rank metrics by root-cause likelihood
using domain-specific prompting.

Pipeline:
  1. Score metrics via RobustScaler z-score
  2. Select candidates via hybrid selection (statistical + early onset + state change)
  3. Build anomaly summary text for LLM prompt
  4. Query LLM (GPT-4o-mini) with domain-specific system prompt + structured user prompt
  5. Backfill any metrics the LLM omitted using heuristic ranking
  6. Cache result for reproducibility
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable

import pandas as pd

from method.algorithms.rca.base import RCAAdapter
from method.algorithms.rca.candidate_selection import (
    build_anomaly_summary,
    score_metrics,
    select_candidates,
)


def build_system_prompt(level: str, domain_phrase: str, domain_context: str) -> str:
    """Build the LLM system prompt for one RCA call.

    Two branches, matching the paper's two DK conditions:

      ``level == "none"``  → domain-neutral framing. Used for the no-DK rows
          in Tables 5 / 6. The LLM is told it is an expert in RCA over
          time-series anomalies but is **not** told which domain the data
          comes from (HVAC, water-treatment, microservice, etc.).

      otherwise            → dataset-specific framing. The LLM is told the
          domain via ``{domain_phrase}`` (a complete noun phrase like
          ``"HVAC rooftop-unit systems"`` or
          ``"an Online Boutique e-commerce microservice platform"``) and is
          handed the ``{domain_context}`` document (the Light variant used
          in the paper) as operational documentation.
    """
    if level == "none" or not domain_context.strip():
        return (
            "You are an expert in root cause analysis of complex systems "
            "based on time-series anomaly evidence."
        )
    return (
        f"You are an expert in {domain_phrase} and root cause analysis. "
        "Use the following operational system documentation to inform "
        f"your reasoning:\n\n{domain_context}"
    )


def build_hybrid_clean_user_prompt(anomaly_summary: str) -> str:
    """Canonical hybrid_clean user prompt — shared across all datasets.

    Wraps the per-scenario ``anomaly_summary`` (already rendered with
    magnitude / onset / state-change evidence) in a dataset-agnostic
    framing that asks the LLM for a reasoning trace and a ranked list.
    """
    return (
        "A fault has been detected. The following items are candidates for "
        "root cause analysis, ranked by deviation from baseline behaviour. "
        "Each line shows the item name, its deviation magnitude, when it "
        "first deviated, and its before/after values.\n\n"
        f"{anomaly_summary}\n\n"
        "Respond with a single JSON object containing BOTH a reasoning trace\n"
        "and the ranked list:\n\n"
        "{\n"
        "  \"reasoning\": \"<your detailed step-by-step analysis identifying "
        "the root cause and explaining why each top-ranked item was chosen "
        "over others>\",\n"
        "  \"ranked\": [\"item1\", \"item2\", ...]\n"
        "}\n\n"
        "Only include items from the provided list. Output only the JSON."
    )


from method.datasets.base import FaultScenario

# Optional Langfuse tracing (no-op if not configured)
try:
    from langfuse import get_client, observe
    _LANGFUSE = get_client() if os.environ.get("LANGFUSE_PUBLIC_KEY") else None
except Exception:
    _LANGFUSE = None
    def observe(**kwargs):
        def _decorator(fn):
            return fn
        return _decorator


def log_run_summary(
    session_id: str | None,
    run_name: str | None,
    name_suffix: str,
    metrics: dict,
    metadata: dict | None = None,
    tags: list[str] | None = None,
) -> None:
    """Emit a Langfuse 'summary' trace under the same session as the per-
    scenario traces. The trace's output is the aggregate metrics dict so it
    appears in the Langfuse UI as the last row of the session.

    Safe to call when Langfuse is not configured — becomes a no-op.
    """
    if _LANGFUSE is None:
        return
    try:
        trace_name = f"summary::{name_suffix}"
        with _LANGFUSE.start_as_current_span(
            name=trace_name,
            input={"summary_for": name_suffix, **(metadata or {})},
        ) as span:
            span.update(output=metrics)
            update_kwargs = dict(
                name=trace_name,
                input={"summary_for": name_suffix},
                output=metrics,
                metadata={**(metadata or {}), "metrics": metrics, "run_name": run_name},
                tags=["summary", *(tags or [])],
            )
            if session_id:
                update_kwargs["session_id"] = session_id
            _LANGFUSE.update_current_trace(**update_kwargs)
    except Exception as e:
        print(f"    [Langfuse] summary log failed: {e}")


class LLMRankingAdapter(RCAAdapter):
    """LLM-based RCA via hybrid candidate selection + domain-specific prompting.

    Parameters
    ----------
    system_prompt : str
        System message providing domain expertise context to the LLM.
    user_prompt_builder : callable
        Function(scenario, anomaly_summary, candidate_names) -> str
        that builds the user prompt. This allows dataset-specific prompting.
    cache_dir : Path
        Directory for caching LLM responses.
    model : str
        OpenAI model name.
    top_n : int
        Number of top candidates to include in the anomaly summary.
    time_unit : str
        Time unit for the anomaly summary (e.g., "s", "min").
    client : object or None
        OpenAI client instance. If None, one will be created on first call.
    """

    requires_graph = False

    def __init__(
        self,
        system_prompt: str,
        user_prompt_builder: Callable[[FaultScenario, str, list[str]], str],
        cache_dir: Path,
        model: str = "gpt-4o-mini",
        top_n: int = 15,
        time_unit: str = "s",
        client=None,
        extra_metadata: dict | None = None,
        extra_tags: list[str] | None = None,
        session_id: str | None = None,
        run_name: str | None = None,
    ):
        self.system_prompt = system_prompt
        self.user_prompt_builder = user_prompt_builder
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.top_n = top_n
        self.time_unit = time_unit
        self.client = client
        self.extra_metadata = extra_metadata or {}
        self.extra_tags = extra_tags or []
        self.session_id = session_id
        self.run_name = run_name

    def _score_trace(self, ranked: list[str], scenario: FaultScenario) -> None:
        """Attach hit@k scores and output to the current Langfuse trace."""
        if _LANGFUSE is None:
            return
        try:
            gt = scenario.ground_truth_causes
            # If the scenario was abstracted with preserve_truth=True, ``ranked``
            # contains opaque tokens (a, b, c, …) but ``gt`` is in real-name
            # space. Map ``ranked`` back through the reverse map before
            # comparing so hit@k reflects the actual ranking quality.
            reverse_map = scenario.metadata.get("abstract_reverse_map") if scenario.metadata else None
            ranked_for_eval = (
                [reverse_map.get(r, r) for r in ranked] if reverse_map else ranked
            )
            hit_1 = 1 if any(g in ranked_for_eval[:1] for g in gt) else 0
            hit_3 = 1 if any(g in ranked_for_eval[:3] for g in gt) else 0
            hit_5 = 1 if any(g in ranked_for_eval[:5] for g in gt) else 0
            gt_pos = next(
                (i + 1 for i, m in enumerate(ranked_for_eval) if m in gt),
                -1,
            )
            _LANGFUSE.update_current_trace(
                output={
                    "ranked_top10": ranked[:10],
                    "gt_position": gt_pos,
                },
            )
            _LANGFUSE.score_current_trace(name="hit@1", value=hit_1)
            _LANGFUSE.score_current_trace(name="hit@3", value=hit_3)
            _LANGFUSE.score_current_trace(name="hit@5", value=hit_5)
            _LANGFUSE.score_current_trace(
                name="reciprocal_rank",
                value=(1.0 / gt_pos) if gt_pos > 0 else 0.0,
            )
        except Exception as e:
            print(f"    [Langfuse] trace finalize error: {e}")

    @observe(name="rca_one_shot_predict")
    def predict(
        self,
        scenario: FaultScenario,
        graph: pd.DataFrame | None = None,
    ) -> list[str]:
        anomaly_summary = build_anomaly_summary(
            scenario, top_n=self.top_n, time_unit=self.time_unit,
        )
        candidates = select_candidates(
            scenario, top_n_score=self.top_n, top_n_early=5, top_n_state=5,
        )
        candidate_names = [m for m, _, _ in candidates]

        # Annotate Langfuse trace with scenario metadata (not sent to LLM)
        if _LANGFUSE is not None:
            try:
                trace_metadata = {
                    "model": self.model,
                    "top_n": self.top_n,
                    "time_unit": self.time_unit,
                    "run_name": self.run_name,
                    # System prompt is recorded so an auditor can confirm the
                    # exact DK level / domain identity used for this trace.
                    "system_prompt": self.system_prompt,
                    # GT stored as metadata only — NEVER sent to the LLM.
                    "ground_truth_eval_only": scenario.ground_truth_causes,
                    **self.extra_metadata,
                }
                trace_tags = [
                    "rca-one-shot",
                    self.model,
                    scenario.scenario_id,
                    *self.extra_tags,
                ]
                if self.run_name:
                    trace_tags.append(f"run={self.run_name}")
                trace_kwargs = dict(
                    name=f"rca_one_shot::{scenario.scenario_id}",
                    input={
                        "scenario_id": scenario.scenario_id,
                        "n_candidates": len(candidate_names),
                        "system_prompt": self.system_prompt,
                    },
                    metadata=trace_metadata,
                    tags=trace_tags,
                )
                if self.session_id:
                    trace_kwargs["session_id"] = self.session_id
                _LANGFUSE.update_current_trace(**trace_kwargs)
            except Exception:
                pass

        # Check cache
        key = hashlib.md5(
            (scenario.scenario_id + anomaly_summary).encode()
        ).hexdigest()
        cache_file = self.cache_dir / f"rank_{key}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                cached = json.load(f)
            if isinstance(cached, list):
                ranked = cached
            else:
                ranked = cached.get("ranked", [])
            self._score_trace(ranked, scenario)
            return ranked

        # Heuristic fallback order
        scored = score_metrics(scenario)
        heuristic_ranked = candidate_names + [
            m for m, _ in scored if m not in set(candidate_names)
        ]

        if os.environ.get("DRY_RUN", "0") == "1":
            return heuristic_ranked

        # Build prompts
        user_prompt = self.user_prompt_builder(
            scenario, anomaly_summary, candidate_names,
        )

        # Call LLM
        try:
            if self.client is None:
                from openai import OpenAI
                self.client = OpenAI()

            llm_messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # GPT-5 family + o-series are reasoning models that reject
            # `temperature` and require `max_completion_tokens` (not max_tokens).
            is_reasoning = self.model.startswith(("gpt-5", "o1", "o3"))
            if is_reasoning:
                call_kwargs = dict(max_completion_tokens=8000, reasoning_effort="medium")
            else:
                call_kwargs = dict(temperature=0, max_tokens=4096)

            if _LANGFUSE is not None:
                with _LANGFUSE.start_as_current_generation(
                    name="llm_one_shot_call",
                    model=self.model,
                    input=llm_messages,
                    model_parameters={k: str(v) for k, v in call_kwargs.items()},
                ) as gen:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=llm_messages,
                        **call_kwargs,
                    )
                    content = response.choices[0].message.content
                    gen.update(
                        output=content,
                        usage_details={
                            "input": response.usage.prompt_tokens if response.usage else 0,
                            "output": response.usage.completion_tokens if response.usage else 0,
                            "total": response.usage.total_tokens if response.usage else 0,
                        } if response.usage else None,
                    )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=llm_messages,
                    **call_kwargs,
                )
                content = response.choices[0].message.content
            # Capture diagnostic metadata from the response (best-effort).
            try:
                finish_reason = response.choices[0].finish_reason
            except Exception:
                finish_reason = ""
            usage = getattr(response, "usage", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            reasoning_tokens = getattr(
                getattr(usage, "completion_tokens_details", None),
                "reasoning_tokens",
                None,
            )
            # Robust JSON extraction for the ranked list.
            import re
            ranked = None
            arr_match = re.search(r'"ranked"\s*:\s*\[([^\]]*)\]', content, re.DOTALL)
            if arr_match:
                ranked = re.findall(r'"([^"]+)"', arr_match.group(1))
            # Try to pull the LLM's narrative reasoning, if present.
            reasoning_text = ""
            try:
                parsed_obj = json.loads(content)
                if not ranked:
                    ranked = parsed_obj.get("ranked", [])
                # Common keys the prompt may have requested for the chain-of-thought.
                for k in ("reasoning", "step_by_step_reasoning", "rationale", "explanation"):
                    val = parsed_obj.get(k)
                    if val:
                        reasoning_text = val if isinstance(val, str) else json.dumps(val)
                        break
            except (json.JSONDecodeError, AttributeError, TypeError):
                # Final fallback: regex-extract a "reasoning" field even if the
                # JSON didn't fully parse.
                rmatch = re.search(
                    r'"(?:reasoning|step_by_step_reasoning|rationale|explanation)"\s*:\s*"((?:[^"\\]|\\.)*)"',
                    content, re.DOTALL,
                )
                if rmatch:
                    reasoning_text = rmatch.group(1)
                if not ranked:
                    ranked = []
            # Backfill metrics the LLM omitted
            ranked_set = set(ranked)
            ranked += [m for m in heuristic_ranked if m not in ranked_set]
        except Exception as e:
            print(f"    [LLM] error: {e}, using heuristic fallback")
            ranked = heuristic_ranked
            content = ""
            reasoning_text = ""
            finish_reason = "exception"
            completion_tokens = None
            prompt_tokens = None
            reasoning_tokens = None

        # Cache
        cache_data = {
            "ranked": ranked,
            "scenario_id": scenario.scenario_id,
            "model": self.model,
            "user_prompt": user_prompt,
            "anomaly_summary": anomaly_summary,
            # New: LLM reasoning trace + usage metadata for auditability.
            "reasoning": reasoning_text,
            "raw_response": content,
            "finish_reason": finish_reason,
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "reasoning_tokens": reasoning_tokens,
        }
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)

        self._score_trace(ranked, scenario)

        return ranked
