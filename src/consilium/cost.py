"""Cost tracking for council calls.

Tracks input/output tokens per call and estimates USD cost based on
published pricing. Prices are approximate — update ``MODEL_PRICING``
as new models are released.
"""

from __future__ import annotations

import dataclasses
from typing import Dict


# Pricing per 1M tokens (input, output) in USD.
# Updated March 2026. See provider pricing pages for current rates.
MODEL_PRICING: Dict[str, tuple[float, float]] = {
    # Anthropic — Claude 4.6 (Feb 2026)
    "claude-opus-4-6": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    # Anthropic — Claude 4.5 (previous gen)
    "claude-haiku-4-5-20251001": (0.80, 4.0),
    "claude-sonnet-4-5-20250514": (3.0, 15.0),
    "claude-opus-4-5": (15.0, 75.0),
    # Google — Gemini 3.x (March 2026)
    "gemini-3.1-pro-preview": (1.25, 10.0),
    "gemini-3-flash-preview": (0.15, 0.60),
    # Google — Gemini 2.5 (stable GA)
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.5-pro": (1.25, 10.0),
    # OpenAI — GPT-5.x series (current)
    "gpt-5.2": (2.0, 8.0),
    "gpt-5.2-pro": (10.0, 40.0),
    "gpt-5": (2.0, 8.0),
    "gpt-5-mini": (0.4, 1.6),
    # OpenAI — GPT-4.x (non-reasoning)
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4.1-nano": (0.1, 0.4),
    # OpenAI — reasoning
    "o3": (2.0, 8.0),
    "o4-mini": (1.1, 4.4),
}

# Fallback when model is not in the pricing table.
DEFAULT_PRICING = (5.0, 15.0)  # conservative estimate


@dataclasses.dataclass
class TokenUsage:
    """Token counts for a single API call."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        """Estimated cost in USD."""
        input_rate, output_rate = MODEL_PRICING.get(self.model, DEFAULT_PRICING)
        return (
            self.input_tokens * input_rate + self.output_tokens * output_rate
        ) / 1_000_000


@dataclasses.dataclass
class CostTracker:
    """Accumulates :class:`TokenUsage` across multiple calls."""

    usages: list[TokenUsage] = dataclasses.field(default_factory=list)

    def record(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> TokenUsage:
        """Record a single API call and return the usage entry."""
        usage = TokenUsage(
            model=model, input_tokens=input_tokens, output_tokens=output_tokens
        )
        self.usages.append(usage)
        return usage

    @property
    def total_cost(self) -> float:
        return sum(u.cost_usd for u in self.usages)

    @property
    def total_input_tokens(self) -> int:
        return sum(u.input_tokens for u in self.usages)

    @property
    def total_output_tokens(self) -> int:
        return sum(u.output_tokens for u in self.usages)

    def breakdown_by_model(self) -> Dict[str, float]:
        """Return a dict mapping model name to total cost for that model."""
        out: Dict[str, float] = {}
        for u in self.usages:
            out[u.model] = out.get(u.model, 0.0) + u.cost_usd
        return out

    def summary(self) -> str:
        """Pretty-print cost summary."""
        lines = ["Cost breakdown:"]
        for model, cost in sorted(self.breakdown_by_model().items()):
            lines.append(f"  {model}: ${cost:.4f}")
        lines.append(f"  TOTAL: ${self.total_cost:.4f}")
        lines.append(
            f"  Tokens: {self.total_input_tokens:,} in"
            f" / {self.total_output_tokens:,} out"
        )
        return "\n".join(lines)

    def exceeds_budget(self, budget: float | None) -> bool:
        """Return True if total cost exceeds the given budget (USD)."""
        if budget is None:
            return False
        return self.total_cost >= budget
