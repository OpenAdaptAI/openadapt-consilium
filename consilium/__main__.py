"""CLI entry point for consilium.

Usage::

    consilium "What color is the sky?"
    consilium "Describe this screenshot" --image screenshot.png
    consilium "Hello" --models gpt-4o,claude-sonnet-4-5 --no-review
    consilium "Plan a trip" --budget 0.50
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap

from consilium.core import LLMCouncil
from consilium.providers import DEFAULT_CHAIRMAN, DEFAULT_MODELS


def _read_image(path: str) -> bytes:
    """Read an image file and return raw bytes."""
    with open(path, "rb") as f:
        return f.read()


def _pretty_print(result, skip_review: bool) -> None:
    """Pretty-print a CouncilResult to stdout."""
    separator = "=" * 72

    # Stage 1
    print(f"\n{separator}")
    print("STAGE 1: INDIVIDUAL RESPONSES")
    print(separator)
    for i, resp in enumerate(result.individual_responses):
        print(f"\n--- {resp.model} ({resp.latency_seconds:.1f}s) ---")
        print(textwrap.fill(resp.text, width=80))

    # Stage 2
    if not skip_review and result.reviews:
        print(f"\n{separator}")
        print("STAGE 2: REVIEWS")
        print(separator)
        for i, rev in enumerate(result.reviews):
            print(f"\n--- Review by {rev.reviewer_model} ({rev.latency_seconds:.1f}s) ---")
            print(textwrap.fill(rev.review_text, width=80))

    # Stage 3
    if not skip_review and result.reviews:
        print(f"\n{separator}")
        print("STAGE 3: CHAIRMAN'S FINAL ANSWER")
        print(separator)
        print()
        print(textwrap.fill(result.final_answer, width=80))

    # Cost
    print(f"\n{separator}")
    print(result.cost_summary())
    print(f"Total latency: {result.total_latency_seconds:.1f}s")
    print(separator)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="consilium",
        description="Query multiple LLMs, cross-review, and synthesize the best answer.",
    )
    parser.add_argument("prompt", help="The question or prompt to send.")
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help=f"Comma-separated model identifiers. Default: {','.join(DEFAULT_MODELS)}",
    )
    parser.add_argument(
        "--chairman",
        default=DEFAULT_CHAIRMAN,
        help=f"Chairman model. Default: {DEFAULT_CHAIRMAN}",
    )
    parser.add_argument("--image", action="append", dest="images", help="Path to an image file.")
    parser.add_argument("--budget", type=float, default=None, help="Max spend in USD.")
    parser.add_argument(
        "--no-review",
        action="store_true",
        help="Skip Stages 2-3; just show individual responses.",
    )
    parser.add_argument("--system", default=None, help="System prompt for all models.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of pretty-printed text.",
    )

    args = parser.parse_args(argv)

    # Parse models
    model_list = [m.strip() for m in args.models.split(",") if m.strip()]

    # Load images
    image_bytes = None
    if args.images:
        image_bytes = [_read_image(p) for p in args.images]

    council = LLMCouncil(models=model_list, chairman=args.chairman)
    result = council.ask(
        args.prompt,
        images=image_bytes,
        budget=args.budget,
        skip_review=args.no_review,
        system=args.system,
    )

    if args.json:
        from consilium.sdk import _result_to_dict

        print(json.dumps(_result_to_dict(result), indent=2))
    else:
        _pretty_print(result, skip_review=args.no_review)


if __name__ == "__main__":
    main()
