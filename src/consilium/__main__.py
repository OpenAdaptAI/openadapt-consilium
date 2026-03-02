"""CLI entry point for Consilium.

Usage::

    consilium "What color is the sky?"
    consilium "Describe this screenshot" --image screenshot.png
    consilium "Hello" --models gpt-5.2,claude-sonnet-4-6 --no-review
    consilium "Plan a trip" --budget 0.50
    consilium "Compare approaches" --json
"""

from __future__ import annotations

import argparse
import json
import textwrap

from consilium.core import Council
from consilium.providers import DEFAULT_CHAIRMAN, get_default_models


def _read_image(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _pretty_print(result, skip_review: bool) -> None:
    separator = "=" * 72

    # Stage 1
    print(f"\n{separator}")
    print("STAGE 1: INDIVIDUAL RESPONSES")
    print(separator)
    for resp in result.individual_responses:
        print(f"\n--- {resp.model} ({resp.latency_seconds:.1f}s) ---")
        print(textwrap.fill(resp.text, width=80))

    # Stage 2
    if not skip_review and result.reviews:
        print(f"\n{separator}")
        print("STAGE 2: REVIEWS")
        print(separator)
        for rev in result.reviews:
            print(
                f"\n--- Review by {rev.reviewer_model}"
                f" ({rev.latency_seconds:.1f}s) ---"
            )
            print(textwrap.fill(rev.review_text, width=80))

    # Stage 3
    if not skip_review and result.reviews:
        print(f"\n{separator}")
        print("STAGE 3: CHAIRMAN'S SYNTHESIS")
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
        description=(
            "Query multiple LLMs, cross-review responses, and synthesize "
            "the best answer."
        ),
    )
    parser.add_argument("prompt", help="The question or prompt to send.")
    parser.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated model identifiers. "
            "Default: auto-detected latest flagship from each provider."
        ),
    )
    parser.add_argument(
        "--chairman",
        default=DEFAULT_CHAIRMAN,
        help=f"Chairman model for synthesis. Default: {DEFAULT_CHAIRMAN}",
    )
    parser.add_argument(
        "--image",
        action="append",
        dest="images",
        help="Path to an image file (can be repeated).",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Max spend in USD.",
    )
    parser.add_argument(
        "--no-review",
        action="store_true",
        help="Skip Stages 2-3; just show individual responses.",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="System prompt for all models.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of pretty-printed text.",
    )

    args = parser.parse_args(argv)

    if args.models is not None:
        model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        model_list = get_default_models()

    image_bytes = None
    if args.images:
        image_bytes = [_read_image(p) for p in args.images]

    council = Council(models=model_list, chairman=args.chairman)
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
