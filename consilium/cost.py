"""Cost tracking for LLM Council calls.

Tracks input/output tokens per call and estimates USD cost based on
published pricing.  Prices are approximate and may drift -- update the
``MODEL_PRICING`` dict as needed.
"""

from __future__ import annotations

import dataclasses
from typing import Dict


# Pricing per 1M tokens (input, output) in USD.
# Keep sorted alphabetically for easy scanning.
MODEL_PRICING: Dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-4": (15.0, 75.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-opus-4-6": (15.0, 75.0),
    # Google
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-3.5-pro": (1.25, 10.0),
    # OpenAI
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-5.3": (2.0, 8.0),
    "o3-mini": (1.1, 4.4),
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
        return (self.input_tokens * input_rate + self.output_tokens * output_rate) / 1_000_000


@dataclasses.dataclass
class CostTracker:
    """Accumulates :class:`TokenUsage` across multiple calls."""

    usages: list[TokenUsage] = dataclasses.field(default_factory=list)

    def record(self, model: str, input_tokens: int, output_tokens: int) -> TokenUsage:
        """Record a single API call and return the usage entry."""
        usage = TokenUsage(model=model, input_tokens=input_tokens, output_tokens=output_tokens)
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
        lines.append(f"  Tokens: {self.total_input_tokens} in / {self.total_output_tokens} out")
        return "\n".join(lines)

    def exceeds_budget(self, budget: float | None) -> bool:
        """Return True if total cost exceeds the given budget (USD)."""
        if budget is None:
            return False
        return self.total_cost >= budget
