"""consilium -- Query multiple LLMs, cross-review, and synthesize the best answer.

Inspired by Karpathy's llm-council (https://github.com/karpathy/llm-council).

Three stages:
  1. First Opinions -- send prompt to all models in parallel
  2. Review -- each model reviews/ranks others' anonymized responses
  3. Final Response -- chairman synthesizes the best answer

Usage::

    from consilium import LLMCouncil

    council = LLMCouncil()
    result = council.ask("What color is the sky?")
    print(result.final_answer)
"""

from consilium.core import CouncilResult, LLMCouncil
from consilium.sdk import council_query

__all__ = ["LLMCouncil", "CouncilResult", "council_query"]
