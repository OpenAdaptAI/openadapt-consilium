"""Tests for the Consilium library.

All API calls are mocked — no real API keys or network access needed.
"""

from __future__ import annotations

import json
import time
from unittest import mock

import pytest

from consilium.cost import CostTracker, TokenUsage
from consilium.core import (
    Council,
    CouncilResult,
    IndividualResponse,
    _anonymize_responses,
)
from consilium.providers import (
    ProviderConfig,
    parse_model_string,
    query_model,
)
from consilium.sdk import council_query, _result_to_dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_query(
    config: ProviderConfig,
    prompt: str,
    images=None,
    system=None,
    json_schema=None,
) -> tuple[str, TokenUsage]:
    text = f"Response from {config.model}: This is a test answer."
    usage = TokenUsage(model=config.model, input_tokens=100, output_tokens=50)
    return text, usage


def _slow_fake_query(
    config: ProviderConfig,
    prompt: str,
    images=None,
    system=None,
    json_schema=None,
) -> tuple[str, TokenUsage]:
    time.sleep(0.05)
    return _fake_query(config, prompt, images, system, json_schema)


@pytest.fixture
def mock_query():
    with mock.patch(
        "consilium.core.query_model", side_effect=_fake_query
    ):
        yield


@pytest.fixture
def mock_query_slow():
    with mock.patch(
        "consilium.core.query_model", side_effect=_slow_fake_query
    ):
        yield


@pytest.fixture
def sample_configs() -> list[ProviderConfig]:
    return [
        ProviderConfig(provider="openai", model="gpt-5.2"),
        ProviderConfig(provider="anthropic", model="claude-sonnet-4-6"),
        ProviderConfig(provider="google", model="gemini-2.5-pro"),
    ]


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------


class TestCostTracking:
    def test_token_usage_cost(self):
        usage = TokenUsage(
            model="gpt-5.2", input_tokens=1_000_000, output_tokens=1_000_000
        )
        # gpt-5.2: $2.00/M input, $8.00/M output = $10.00
        assert abs(usage.cost_usd - 10.00) < 0.01

    def test_token_usage_unknown_model(self):
        usage = TokenUsage(
            model="unknown-model",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )
        assert abs(usage.cost_usd - 20.00) < 0.01

    def test_cost_tracker_record(self):
        tracker = CostTracker()
        tracker.record("gpt-5.2", 500, 100)
        tracker.record("gpt-5.2", 300, 200)
        assert len(tracker.usages) == 2
        assert tracker.total_input_tokens == 800
        assert tracker.total_output_tokens == 300

    def test_cost_tracker_breakdown(self):
        tracker = CostTracker()
        tracker.record("gpt-5.2", 1000, 500)
        tracker.record("claude-sonnet-4-6", 1000, 500)
        breakdown = tracker.breakdown_by_model()
        assert "gpt-5.2" in breakdown
        assert "claude-sonnet-4-6" in breakdown

    def test_cost_tracker_exceeds_budget(self):
        tracker = CostTracker()
        assert not tracker.exceeds_budget(1.0)
        assert not tracker.exceeds_budget(None)
        tracker.record("gpt-5.2", 100_000, 100_000)
        assert tracker.exceeds_budget(0.001)

    def test_cost_tracker_summary_format(self):
        tracker = CostTracker()
        tracker.record("gpt-5.2", 1000, 500)
        summary = tracker.summary()
        assert "Cost breakdown:" in summary
        assert "gpt-5.2" in summary
        assert "TOTAL:" in summary
        assert "Tokens:" in summary


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


class TestProviders:
    def test_parse_model_with_slash(self):
        cfg = parse_model_string("openai/gpt-5.2")
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-5.2"

    def test_parse_model_alias(self):
        cfg = parse_model_string("gpt-5.2")
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-5.2"

    def test_parse_model_anthropic_alias(self):
        cfg = parse_model_string("claude-sonnet-4-6")
        assert cfg.provider == "anthropic"
        assert cfg.model == "claude-sonnet-4-6"

    def test_parse_model_google_alias(self):
        cfg = parse_model_string("gemini-2.5-pro")
        assert cfg.provider == "google"
        assert cfg.model == "gemini-2.5-pro"

    def test_parse_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Cannot parse model"):
            parse_model_string("totally-unknown")

    def test_unknown_provider_raises(self):
        cfg = ProviderConfig(provider="fakecloud", model="v1")
        with pytest.raises(ValueError, match="Unknown provider"):
            query_model(cfg, "hello")

    def test_display_name(self):
        cfg = ProviderConfig(provider="openai", model="gpt-5.2")
        assert cfg.display_name == "openai/gpt-5.2"


# ---------------------------------------------------------------------------
# Anonymization
# ---------------------------------------------------------------------------


class TestAnonymization:
    def test_anonymize_basic(self):
        responses = [
            IndividualResponse(
                model="openai/gpt-5.2",
                text="Answer one",
                latency_seconds=1.0,
                usage=TokenUsage(model="gpt-5.2"),
            ),
            IndividualResponse(
                model="anthropic/claude-sonnet-4-6",
                text="Answer two",
                latency_seconds=1.0,
                usage=TokenUsage(model="claude-sonnet-4-6"),
            ),
        ]
        result = _anonymize_responses(responses)
        assert "gpt-5.2" not in result
        assert "claude" not in result
        assert "Response A" in result
        assert "Response B" in result
        assert "Answer one" in result
        assert "Answer two" in result

    def test_anonymize_preserves_order(self):
        responses = [
            IndividualResponse(
                model=f"model-{i}",
                text=f"Text {i}",
                latency_seconds=0.0,
                usage=TokenUsage(model=f"model-{i}"),
            )
            for i in range(5)
        ]
        result = _anonymize_responses(responses)
        assert result.index("Response A") < result.index("Response B")
        assert result.index("Response B") < result.index("Response C")


# ---------------------------------------------------------------------------
# Core council
# ---------------------------------------------------------------------------


class TestCouncil:
    def test_full_pipeline(self, mock_query, sample_configs):
        council = Council(models=sample_configs, chairman=sample_configs[1])
        result = council.ask("What is 2+2?")

        assert isinstance(result, CouncilResult)
        assert len(result.individual_responses) == 3
        assert len(result.reviews) == 3
        assert result.final_answer
        assert result.total_latency_seconds >= 0
        assert result.total_cost >= 0

    def test_skip_review(self, mock_query, sample_configs):
        council = Council(models=sample_configs, chairman=sample_configs[0])
        result = council.ask("Test", skip_review=True)

        assert len(result.individual_responses) == 3
        assert len(result.reviews) == 0
        assert result.final_answer

    def test_budget_enforcement_skips_review(self, sample_configs):
        def expensive_query(
            config, prompt, images=None, system=None, json_schema=None
        ):
            return (
                f"Answer from {config.model}",
                TokenUsage(
                    model=config.model,
                    input_tokens=1_000_000,
                    output_tokens=1_000_000,
                ),
            )

        with mock.patch(
            "consilium.core.query_model", side_effect=expensive_query
        ):
            council = Council(
                models=sample_configs, chairman=sample_configs[0]
            )
            result = council.ask("Test", budget=0.001)

        assert len(result.reviews) == 0
        assert result.total_cost > 0.001

    def test_parallel_execution(self, mock_query_slow, sample_configs):
        council = Council(models=sample_configs, chairman=sample_configs[0])
        t0 = time.monotonic()
        result = council.ask("Test", skip_review=True)
        elapsed = time.monotonic() - t0

        assert elapsed < 0.12, f"Queries appear sequential: {elapsed:.3f}s"
        assert len(result.individual_responses) == 3

    def test_error_handling(self, sample_configs):
        def sometimes_fail(
            config, prompt, images=None, system=None, json_schema=None
        ):
            if config.model == "gpt-5.2":
                raise RuntimeError("API timeout")
            return _fake_query(config, prompt, images, system, json_schema)

        with mock.patch(
            "consilium.core.query_model", side_effect=sometimes_fail
        ):
            council = Council(
                models=sample_configs, chairman=sample_configs[1]
            )
            result = council.ask("Test", skip_review=True)

        assert len(result.individual_responses) == 3
        error_resp = [
            r for r in result.individual_responses if "[ERROR:" in r.text
        ]
        assert len(error_resp) == 1
        assert "API timeout" in error_resp[0].text

    def test_image_input_passed_through(self, sample_configs):
        received_images = []

        def capture_images(
            config, prompt, images=None, system=None, json_schema=None
        ):
            received_images.append(images)
            return _fake_query(config, prompt, images, system, json_schema)

        with mock.patch(
            "consilium.core.query_model", side_effect=capture_images
        ):
            council = Council(
                models=sample_configs[:1], chairman=sample_configs[0]
            )
            fake_image = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
            council.ask(
                "Describe this", images=[fake_image], skip_review=True
            )

        assert len(received_images) == 1
        assert received_images[0] == [fake_image]

    def test_with_string_models(self, mock_query):
        council = Council(
            models=["gpt-5.2", "claude-sonnet-4-6"],
            chairman="gpt-5.2",
        )
        result = council.ask("Test", skip_review=True)
        assert len(result.individual_responses) == 2

    def test_cost_summary(self, mock_query, sample_configs):
        council = Council(models=sample_configs, chairman=sample_configs[0])
        result = council.ask("Test", skip_review=True)
        summary = result.cost_summary()
        assert "Cost breakdown:" in summary
        assert "TOTAL:" in summary


# ---------------------------------------------------------------------------
# SDK
# ---------------------------------------------------------------------------


class TestSDK:
    def test_council_query_returns_dict(self, mock_query):
        result = council_query(
            "What is 2+2?",
            models=["gpt-5.2", "claude-sonnet-4-6"],
            skip_review=True,
        )
        assert isinstance(result, dict)
        assert "final_answer" in result
        assert "individual_responses" in result
        assert "reviews" in result
        assert "cost" in result
        assert "total_latency_seconds" in result

    def test_council_query_response_structure(self, mock_query):
        result = council_query(
            "Test", models=["gpt-5.2"], skip_review=True
        )
        resp = result["individual_responses"][0]
        assert "model" in resp
        assert "text" in resp
        assert "latency_seconds" in resp
        assert "input_tokens" in resp
        assert "output_tokens" in resp
        assert "cost_usd" in resp

    def test_council_query_cost_structure(self, mock_query):
        result = council_query("Test", models=["gpt-5.2"], skip_review=True)
        cost = result["cost"]
        assert "breakdown" in cost
        assert "total_usd" in cost
        assert "total_input_tokens" in cost
        assert "total_output_tokens" in cost

    def test_council_query_serializable(self, mock_query):
        result = council_query("Test", models=["gpt-5.2"], skip_review=True)
        serialized = json.dumps(result)
        assert isinstance(serialized, str)
        round_tripped = json.loads(serialized)
        assert round_tripped["final_answer"] == result["final_answer"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_cli_basic(self, mock_query, capsys):
        from consilium.__main__ import main

        main(["What is 2+2?", "--models", "gpt-5.2", "--no-review"])
        captured = capsys.readouterr()
        assert "INDIVIDUAL RESPONSES" in captured.out
        assert "Cost breakdown:" in captured.out

    def test_cli_json_output(self, mock_query, capsys):
        from consilium.__main__ import main

        main(["Test", "--models", "gpt-5.2", "--no-review", "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "final_answer" in data

    def test_cli_budget(self, mock_query, capsys):
        from consilium.__main__ import main

        main(["Test", "--models", "gpt-5.2", "--budget", "0.50"])
        captured = capsys.readouterr()
        assert "Cost breakdown:" in captured.out

    def test_cli_with_review(self, mock_query, capsys):
        from consilium.__main__ import main

        main(
            [
                "Test",
                "--models",
                "gpt-5.2,claude-sonnet-4-6",
            ]
        )
        captured = capsys.readouterr()
        assert "STAGE 1" in captured.out
        assert "STAGE 2" in captured.out
        assert "STAGE 3" in captured.out


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_three_stage_flow(self, mock_query):
        council = Council(
            models=["gpt-5.2", "claude-sonnet-4-6", "gemini-2.5-pro"],
            chairman="claude-sonnet-4-6",
        )
        result = council.ask("Explain quantum computing in one paragraph.")

        assert len(result.individual_responses) == 3
        assert len(result.reviews) == 3
        assert len(result.final_answer) > 0
        # 3 (stage1) + 3 (stage2) + 1 (stage3) = 7 calls
        assert len(result.cost_tracker.usages) == 7
        assert result.total_cost > 0

    def test_single_model_council(self, mock_query):
        council = Council(models=["gpt-5.2"], chairman="gpt-5.2")
        result = council.ask("Test")

        assert len(result.individual_responses) == 1
        assert len(result.reviews) == 1
        assert result.final_answer

    def test_result_to_dict_roundtrip(self, mock_query):
        council = Council(models=["gpt-5.2"], chairman="gpt-5.2")
        result = council.ask("Test", skip_review=True)
        d = _result_to_dict(result)
        s = json.dumps(d)
        d2 = json.loads(s)
        assert d2["final_answer"] == d["final_answer"]
        assert d2["cost"]["total_usd"] == d["cost"]["total_usd"]
