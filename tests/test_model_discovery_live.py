"""Live integration tests for model discovery.

All tests are marked ``@pytest.mark.live`` and auto-skipped when the
corresponding API key is absent.  Run with::

    OPENAI_API_KEY=... ANTHROPIC_API_KEY=... GOOGLE_API_KEY=... \
        uv run pytest -m live -v
"""

from __future__ import annotations

import os

import pytest

from consilium.model_registry import (
    _classify_tier,
    _is_openai_chat_model,
    clear_cache,
    get_default_models,
    get_latest,
    list_models,
)

skip_no_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
skip_no_anthropic = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
skip_no_google = pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set",
)

pytestmark = pytest.mark.live


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_cache()
    yield
    clear_cache()


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


@skip_no_openai
@pytest.mark.timeout(30)
class TestOpenAILive:
    def test_list_returns_models(self):
        models = list_models("openai")
        assert len(models) > 0

    def test_only_chat_models(self):
        models = list_models("openai")
        for m in models:
            assert _is_openai_chat_model(m.id), f"non-chat model leaked: {m.id}"

    def test_has_gpt_models(self):
        models = list_models("openai")
        gpt_ids = [m.id for m in models if m.id.startswith("gpt-")]
        assert len(gpt_ids) > 0, "expected at least one gpt-* model"

    def test_all_classify_into_tiers(self):
        models = list_models("openai")
        for m in models:
            tier = _classify_tier("openai", m.id)
            assert tier is not None, f"{m.id} did not classify into any tier"

    def test_get_latest_flagship(self):
        model_id = get_latest("openai", "flagship")
        assert model_id
        assert "gpt" in model_id.lower()


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


@skip_no_anthropic
@pytest.mark.timeout(30)
class TestAnthropicLive:
    def test_list_returns_models(self):
        models = list_models("anthropic")
        assert len(models) > 0

    def test_only_claude_models(self):
        models = list_models("anthropic")
        for m in models:
            assert m.id.startswith("claude-"), f"non-Claude model leaked: {m.id}"

    def test_has_flagship(self):
        models = list_models("anthropic")
        flagships = [
            m for m in models if _classify_tier("anthropic", m.id) == "flagship"
        ]
        assert len(flagships) > 0, "expected at least one flagship Claude model"

    def test_display_names_populated(self):
        models = list_models("anthropic")
        for m in models:
            assert m.display_name, f"{m.id} has empty display_name"


# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------


@skip_no_google
@pytest.mark.timeout(30)
class TestGoogleLive:
    def test_list_returns_models(self):
        models = list_models("google")
        assert len(models) > 0

    def test_only_gemini_models(self):
        models = list_models("google")
        for m in models:
            assert "gemini" in m.id.lower(), f"non-Gemini model leaked: {m.id}"

    def test_no_models_prefix(self):
        models = list_models("google")
        for m in models:
            assert not m.id.startswith("models/"), (
                f"models/ prefix not stripped: {m.id}"
            )

    def test_has_pro_and_flash(self):
        models = list_models("google")
        ids = {m.id for m in models}
        has_pro = any("pro" in mid.lower() for mid in ids)
        has_flash = any("flash" in mid.lower() for mid in ids)
        assert has_pro, "expected at least one *pro* model"
        assert has_flash, "expected at least one *flash* model"

    def test_get_latest_works(self):
        model_id = get_latest("google", "flagship")
        assert model_id
        assert "pro" in model_id.lower() or "gemini" in model_id.lower()


# ---------------------------------------------------------------------------
# Cross-provider
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
class TestCrossProviderLive:
    @skip_no_openai
    @skip_no_anthropic
    @skip_no_google
    def test_get_default_models_returns_three(self):
        models = get_default_models()
        assert len(models) == 3
        providers = {m.split("/")[0] for m in models}
        assert providers == {"openai", "anthropic", "google"}

    @skip_no_openai
    def test_caching_works(self):
        first = list_models("openai")
        second = list_models("openai")
        assert first == second
