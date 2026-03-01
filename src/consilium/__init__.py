"""Consilium — multi-LLM council for consensus-driven AI responses.

Inspired by Karpathy's llm-council (https://github.com/karpathy/llm-council).

Three-stage pipeline:
  1. First Opinions — send prompt to all models in parallel
  2. Review — each model reviews/ranks others' anonymized responses
  3. Synthesis — chairman synthesizes the best answer

Usage::

    from consilium import Council

    council = Council()
    result = council.ask("What color is the sky?")
    print(result.final_answer)
    print(result.cost_summary())
"""

from consilium.core import Council, CouncilResult
from consilium.sdk import council_query

__version__ = "0.1.0"
__all__ = ["Council", "CouncilResult", "council_query"]
