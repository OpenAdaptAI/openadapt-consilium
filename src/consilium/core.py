"""Core council implementation.

Three-stage pipeline:
  1. First Opinions — parallel queries to all models
  2. Review — each model reviews anonymized responses from the others
  3. Synthesis — chairman synthesizes the best answer
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import string
import time
from typing import Any, Dict, List

from consilium.cost import CostTracker, TokenUsage
from consilium.providers import (
    DEFAULT_CHAIRMAN,
    DEFAULT_MODELS,
    ProviderConfig,
    parse_model_string,
    query_model,
)


@dataclasses.dataclass
class IndividualResponse:
    """A single model's response from Stage 1."""

    model: str
    text: str
    latency_seconds: float
    usage: TokenUsage


@dataclasses.dataclass
class Review:
    """A single model's review of others' responses from Stage 2."""

    reviewer_model: str
    review_text: str
    latency_seconds: float
    usage: TokenUsage


@dataclasses.dataclass
class CouncilResult:
    """Full result from a council query across all three stages."""

    final_answer: str
    individual_responses: List[IndividualResponse]
    reviews: List[Review]
    cost_tracker: CostTracker
    total_latency_seconds: float

    @property
    def cost_breakdown(self) -> Dict[str, float]:
        return self.cost_tracker.breakdown_by_model()

    @property
    def total_cost(self) -> float:
        return self.cost_tracker.total_cost

    def cost_summary(self) -> str:
        return self.cost_tracker.summary()


_LABELS = list(string.ascii_uppercase)


def _anonymize_responses(responses: List[IndividualResponse]) -> str:
    """Format responses as anonymized text for the review prompt."""
    parts = []
    for i, resp in enumerate(responses):
        label = _LABELS[i] if i < len(_LABELS) else f"Response {i + 1}"
        parts.append(f"--- Response {label} ---\n{resp.text}\n")
    return "\n".join(parts)


class Council:
    """Orchestrates multi-LLM querying, cross-review, and synthesis.

    Args:
        models: List of model strings (e.g. ``["gpt-4.1", "claude-sonnet-4-5-20250514"]``)
            or :class:`ProviderConfig` objects.
        chairman: Model string or config for the synthesis stage.
        max_workers: Max threads for parallel queries.

    Example::

        council = Council()
        result = council.ask("Summarize quantum computing in one paragraph.")
        print(result.final_answer)
        print(result.cost_summary())
    """

    def __init__(
        self,
        models: List[str | ProviderConfig] | None = None,
        chairman: str | ProviderConfig | None = None,
        max_workers: int = 8,
    ) -> None:
        raw_models = models or DEFAULT_MODELS
        self.configs: List[ProviderConfig] = [
            m if isinstance(m, ProviderConfig) else parse_model_string(m)
            for m in raw_models
        ]
        self.chairman_config: ProviderConfig = (
            chairman
            if isinstance(chairman, ProviderConfig)
            else parse_model_string(chairman or DEFAULT_CHAIRMAN)
        )
        self.max_workers = max_workers

    def ask(
        self,
        prompt: str,
        images: list[bytes] | None = None,
        budget: float | None = None,
        system: str | None = None,
        skip_review: bool = False,
        json_schema: dict | None = None,
    ) -> CouncilResult:
        """Run the full council pipeline.

        Args:
            prompt: User query.
            images: Optional images (PNG bytes) to include in each query.
            budget: Max spend in USD. If Stage 1 exceeds budget, Stages 2-3
                are skipped.
            system: Optional system prompt prepended to every call.
            skip_review: If ``True``, skip Stages 2 and 3 entirely.
            json_schema: If set, request structured JSON output.

        Returns:
            :class:`CouncilResult` with all responses, reviews, and the
            synthesized final answer.
        """
        t0 = time.monotonic()
        tracker = CostTracker()

        # Stage 1: First Opinions
        responses = self._stage1(
            prompt,
            images=images,
            system=system,
            tracker=tracker,
            json_schema=json_schema,
        )

        if skip_review or tracker.exceeds_budget(budget):
            best = (
                max(responses, key=lambda r: len(r.text)) if responses else None
            )
            return CouncilResult(
                final_answer=best.text if best else "",
                individual_responses=responses,
                reviews=[],
                cost_tracker=tracker,
                total_latency_seconds=time.monotonic() - t0,
            )

        # Stage 2: Review
        reviews = self._stage2(
            prompt,
            responses,
            images=images,
            system=system,
            tracker=tracker,
        )

        if tracker.exceeds_budget(budget):
            best = (
                max(responses, key=lambda r: len(r.text)) if responses else None
            )
            return CouncilResult(
                final_answer=best.text if best else "",
                individual_responses=responses,
                reviews=reviews,
                cost_tracker=tracker,
                total_latency_seconds=time.monotonic() - t0,
            )

        # Stage 3: Synthesis
        final_answer, _, _ = self._stage3(
            prompt,
            responses,
            reviews,
            images=images,
            system=system,
            tracker=tracker,
        )

        return CouncilResult(
            final_answer=final_answer,
            individual_responses=responses,
            reviews=reviews,
            cost_tracker=tracker,
            total_latency_seconds=time.monotonic() - t0,
        )

    # ------------------------------------------------------------------
    # Internal stages
    # ------------------------------------------------------------------

    def _stage1(
        self,
        prompt: str,
        *,
        images: list[bytes] | None,
        system: str | None,
        tracker: CostTracker,
        json_schema: dict | None = None,
    ) -> List[IndividualResponse]:
        def _query_one(cfg: ProviderConfig) -> IndividualResponse:
            t0 = time.monotonic()
            text, usage = query_model(
                cfg,
                prompt,
                images=images,
                system=system,
                json_schema=json_schema,
            )
            elapsed = time.monotonic() - t0
            tracker.record(cfg.model, usage.input_tokens, usage.output_tokens)
            return IndividualResponse(
                model=cfg.display_name,
                text=text,
                latency_seconds=elapsed,
                usage=usage,
            )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as pool:
            futures = {pool.submit(_query_one, cfg): cfg for cfg in self.configs}
            results = []
            for fut in concurrent.futures.as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as exc:
                    cfg = futures[fut]
                    results.append(
                        IndividualResponse(
                            model=cfg.display_name,
                            text=f"[ERROR: {exc}]",
                            latency_seconds=0.0,
                            usage=TokenUsage(model=cfg.model),
                        )
                    )
        return results

    def _stage2(
        self,
        original_prompt: str,
        responses: List[IndividualResponse],
        *,
        images: list[bytes] | None,
        system: str | None,
        tracker: CostTracker,
    ) -> List[Review]:
        anonymized = _anonymize_responses(responses)
        review_prompt = (
            f"You are reviewing responses to the following question:\n\n"
            f"QUESTION: {original_prompt}\n\n"
            f"Below are anonymized responses from different AI models. "
            f"Please review them. For each response, briefly note its "
            f"strengths and weaknesses. Then rank them from best to worst "
            f"and explain your ranking.\n\n"
            f"{anonymized}"
        )

        def _review_one(cfg: ProviderConfig) -> Review:
            t0 = time.monotonic()
            text, usage = query_model(
                cfg, review_prompt, images=images, system=system
            )
            elapsed = time.monotonic() - t0
            tracker.record(cfg.model, usage.input_tokens, usage.output_tokens)
            return Review(
                reviewer_model=cfg.display_name,
                review_text=text,
                latency_seconds=elapsed,
                usage=usage,
            )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as pool:
            futures = {pool.submit(_review_one, cfg): cfg for cfg in self.configs}
            reviews = []
            for fut in concurrent.futures.as_completed(futures):
                try:
                    reviews.append(fut.result())
                except Exception as exc:
                    cfg = futures[fut]
                    reviews.append(
                        Review(
                            reviewer_model=cfg.display_name,
                            review_text=f"[ERROR: {exc}]",
                            latency_seconds=0.0,
                            usage=TokenUsage(model=cfg.model),
                        )
                    )
        return reviews

    def _stage3(
        self,
        original_prompt: str,
        responses: List[IndividualResponse],
        reviews: List[Review],
        *,
        images: list[bytes] | None,
        system: str | None,
        tracker: CostTracker,
    ) -> tuple[str, TokenUsage, float]:
        anonymized = _anonymize_responses(responses)
        reviews_text = "\n\n".join(
            f"--- Review by Reviewer {i + 1} ---\n{r.review_text}"
            for i, r in enumerate(reviews)
        )

        synthesis_prompt = (
            f"You are the Chairman of an LLM Council. Multiple AI models "
            f"were asked the following question, and then each reviewed "
            f"the others' responses.\n\n"
            f"QUESTION: {original_prompt}\n\n"
            f"RESPONSES:\n{anonymized}\n\n"
            f"REVIEWS:\n{reviews_text}\n\n"
            f"Based on all responses and reviews, synthesize the single "
            f"best answer. Combine the strongest points from each response "
            f"while correcting any errors noted in the reviews. Be concise "
            f"and authoritative."
        )

        t0 = time.monotonic()
        text, usage = query_model(
            self.chairman_config,
            synthesis_prompt,
            images=images,
            system=system,
        )
        elapsed = time.monotonic() - t0
        tracker.record(
            self.chairman_config.model,
            usage.input_tokens,
            usage.output_tokens,
        )
        return text, usage, elapsed
