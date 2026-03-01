"""High-level SDK for programmatic council queries.

Provides a simple function-call interface designed for AI agents and
automated pipelines.

Example::

    from consilium import council_query

    result = council_query(
        "What are the steps to open Notepad on Windows 11?",
        models=["gpt-4o", "claude-sonnet-4-5"],
    )
    print(result["final_answer"])
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from consilium.core import CouncilResult, LLMCouncil
from consilium.providers import ProviderConfig


def council_query(
    question: str,
    *,
    images: list[bytes] | None = None,
    models: list[str] | None = None,
    chairman: str | None = None,
    budget: float | None = None,
    system: str | None = None,
    skip_review: bool = False,
    json_schema: dict | None = None,
) -> Dict[str, Any]:
    """Query the LLM council and return a structured dict result.

    This is the primary programmatic entry point for agents and scripts.

    Args:
        question: The question / prompt to send.
        images: Optional list of image bytes (PNG).
        models: Model identifiers (e.g. ``["gpt-4o", "claude-sonnet-4-5"]``).
        chairman: Chairman model identifier.
        budget: Maximum spend in USD.
        system: Optional system prompt.
        skip_review: If True, skip Stages 2-3.
        json_schema: If set, request structured JSON output from each model.

    Returns:
        Dict with keys:
        - ``final_answer`` (str): Synthesized answer from the chairman.
        - ``individual_responses`` (list[dict]): Each model's response.
        - ``reviews`` (list[dict]): Each model's review (empty if skipped).
        - ``cost`` (dict): Cost breakdown by model and total.
        - ``total_latency_seconds`` (float): Wall-clock time.
    """
    council = LLMCouncil(models=models, chairman=chairman)
    result: CouncilResult = council.ask(
        question,
        images=images,
        budget=budget,
        system=system,
        skip_review=skip_review,
        json_schema=json_schema,
    )
    return _result_to_dict(result)


def _result_to_dict(result: CouncilResult) -> Dict[str, Any]:
    """Serialize a :class:`CouncilResult` into a plain dict."""
    return {
        "final_answer": result.final_answer,
        "individual_responses": [
            {
                "model": r.model,
                "text": r.text,
                "latency_seconds": round(r.latency_seconds, 3),
                "input_tokens": r.usage.input_tokens,
                "output_tokens": r.usage.output_tokens,
                "cost_usd": round(r.usage.cost_usd, 6),
            }
            for r in result.individual_responses
        ],
        "reviews": [
            {
                "reviewer_model": r.reviewer_model,
                "review_text": r.review_text,
                "latency_seconds": round(r.latency_seconds, 3),
                "input_tokens": r.usage.input_tokens,
                "output_tokens": r.usage.output_tokens,
                "cost_usd": round(r.usage.cost_usd, 6),
            }
            for r in result.reviews
        ],
        "cost": {
            "breakdown": result.cost_breakdown,
            "total_usd": round(result.total_cost, 6),
            "total_input_tokens": result.cost_tracker.total_input_tokens,
            "total_output_tokens": result.cost_tracker.total_output_tokens,
        },
        "total_latency_seconds": round(result.total_latency_seconds, 3),
    }
