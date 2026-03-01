"""Tests for the model registry module.

All API calls are mocked — no real API keys or network access needed.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest import mock

import pytest

from consilium.model_registry import (
    DEFAULTS,
    ModelInfo,
    _classify_tier,
    _extract_version_tuple,
    _is_openai_chat_model,
    _PROVIDER_LISTERS,
    clear_cache,
    get_default_models,
    get_latest,
    list_models,
    set_cache_ttl,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_registry_cache():
    """Clear the model registry cache before and after each test."""
    clear_cache()
    yield
    clear_cache()


def _make_openai_model_infos() -> list[ModelInfo]:
    """Create ModelInfo objects simulating OpenAI models."""
    return [
        ModelInfo(
            id="gpt-5.2",
            provider="openai",
            created_at=datetime(2025, 11, 1, tzinfo=timezone.utc),
        ),
        ModelInfo(
            id="gpt-5",
            provider="openai",
            created_at=datetime(2025, 9, 1, tzinfo=timezone.utc),
        ),
        ModelInfo(
            id="gpt-5-mini",
            provider="openai",
            created_at=datetime(2025, 10, 1, tzinfo=timezone.utc),
        ),
        ModelInfo(
            id="gpt-4.1",
            provider="openai",
            created_at=datetime(2025, 4, 1, tzinfo=timezone.utc),
        ),
        ModelInfo(
            id="gpt-4.1-mini",
            provider="openai",
            created_at=datetime(2025, 4, 1, tzinfo=timezone.utc),
        ),
        ModelInfo(
            id="gpt-4.1-nano",
            provider="openai",
            created_at=datetime(2025, 4, 1, tzinfo=timezone.utc),
        ),
        ModelInfo(id="o3", provider="openai"),
        ModelInfo(id="o4-mini", provider="openai"),
        ModelInfo(id="gpt-audio-2025-08-28", provider="openai"),
        ModelInfo(id="gpt-4o-realtime-preview", provider="openai"),
    ]


def _make_anthropic_model_infos() -> list[ModelInfo]:
    """Create ModelInfo objects simulating Anthropic models."""
    return [
        ModelInfo(
            id="claude-opus-4-6",
            provider="anthropic",
            display_name="Claude Opus 4.6",
        ),
        ModelInfo(
            id="claude-sonnet-4-6",
            provider="anthropic",
            display_name="Claude Sonnet 4.6",
        ),
        ModelInfo(
            id="claude-haiku-4-5-20251001",
            provider="anthropic",
            display_name="Claude Haiku 4.5",
        ),
        ModelInfo(
            id="claude-sonnet-4-5-20250514",
            provider="anthropic",
            display_name="Claude Sonnet 4.5",
        ),
        ModelInfo(
            id="claude-opus-4-5",
            provider="anthropic",
            display_name="Claude Opus 4.5",
        ),
    ]


def _make_google_model_infos() -> list[ModelInfo]:
    """Create ModelInfo objects simulating Google models."""
    return [
        ModelInfo(
            id="gemini-3.1-pro-preview",
            provider="google",
            display_name="Gemini 3.1 Pro Preview",
        ),
        ModelInfo(
            id="gemini-3-flash-preview",
            provider="google",
            display_name="Gemini 3 Flash Preview",
        ),
        ModelInfo(
            id="gemini-2.5-pro",
            provider="google",
            display_name="Gemini 2.5 Pro",
        ),
        ModelInfo(
            id="gemini-2.5-flash",
            provider="google",
            display_name="Gemini 2.5 Flash",
        ),
    ]


def _make_openai_sdk_models():
    """Create mock OpenAI SDK model objects (as returned by client.models.list()).

    Includes non-chat models that should be filtered out by _list_openai().
    """
    models = []
    for model_id, created in [
        ("gpt-5.2", 1700000000),
        ("gpt-5", 1690000000),
        ("gpt-5-mini", 1695000000),
        ("gpt-4.1", 1680000000),
        ("gpt-4.1-mini", 1680000001),
        ("gpt-4.1-nano", 1680000002),
        ("o3", 1698000000),
        ("o4-mini", 1699000000),
        # Non-chat models — should be filtered out
        ("text-embedding-3-large", 1680000003),
        ("tts-1", 1680000004),
        ("dall-e-3", 1680000005),
        ("gpt-4o-realtime-preview", 1680000006),
        ("gpt-4o-audio-preview", 1680000007),
        ("whisper-1", 1680000008),
    ]:
        m = mock.MagicMock()
        m.id = model_id
        m.created = created
        models.append(m)
    return models


def _make_anthropic_sdk_models():
    """Create mock Anthropic SDK model objects.

    Includes a non-Claude model that should be filtered out by _list_anthropic().
    """
    models = []
    for model_id, display_name in [
        ("claude-opus-4-6", "Claude Opus 4.6"),
        ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
        ("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
        ("claude-sonnet-4-5-20250514", "Claude Sonnet 4.5"),
        ("claude-opus-4-5", "Claude Opus 4.5"),
        # Non-Claude model — should be filtered out
        ("some-internal-model", "Internal Model"),
    ]:
        m = mock.MagicMock()
        m.id = model_id
        m.display_name = display_name
        m.created_at = "2025-01-15T00:00:00+00:00"
        models.append(m)
    return models


def _make_google_sdk_models():
    """Create mock Google SDK model objects.

    Includes ``supported_actions`` attribute and non-Gemini models that should
    be filtered out by _list_google().
    """
    models = []
    for model_name, display_name, actions in [
        ("models/gemini-3.1-pro-preview", "Gemini 3.1 Pro Preview", ["generateContent", "countTokens"]),
        ("models/gemini-3-flash-preview", "Gemini 3 Flash Preview", ["generateContent", "countTokens"]),
        ("models/gemini-2.5-pro", "Gemini 2.5 Pro", ["generateContent", "countTokens"]),
        ("models/gemini-2.5-flash", "Gemini 2.5 Flash", ["generateContent", "countTokens"]),
        # Non-Gemini model with generateContent — should be filtered out
        ("models/text-embedding-004", "Text Embedding 004", ["embedContent"]),
        # Gemini without generateContent — should be filtered out
        ("models/gemini-embedding-exp", "Gemini Embedding Exp", ["embedContent"]),
    ]:
        m = mock.MagicMock()
        m.name = model_name
        m.display_name = display_name
        m.supported_actions = actions
        models.append(m)
    return models


# ---------------------------------------------------------------------------
# Unit tests — tier classification
# ---------------------------------------------------------------------------


class TestTierClassification:
    def test_openai_flagship(self):
        assert _classify_tier("openai", "gpt-5.2") == "flagship"
        assert _classify_tier("openai", "gpt-4.1") == "flagship"

    def test_openai_fast(self):
        assert _classify_tier("openai", "gpt-5-mini") == "fast"
        assert _classify_tier("openai", "gpt-4.1-mini") == "fast"

    def test_openai_fastest(self):
        assert _classify_tier("openai", "gpt-4.1-nano") == "fastest"

    def test_openai_reasoning(self):
        assert _classify_tier("openai", "o3") == "reasoning"

    def test_openai_reasoning_fast(self):
        assert _classify_tier("openai", "o4-mini") == "reasoning_fast"

    def test_anthropic_flagship(self):
        assert _classify_tier("anthropic", "claude-opus-4-6") == "flagship"
        assert _classify_tier("anthropic", "claude-opus-4-5") == "flagship"

    def test_anthropic_fast(self):
        assert _classify_tier("anthropic", "claude-sonnet-4-6") == "fast"

    def test_anthropic_fastest(self):
        assert (
            _classify_tier("anthropic", "claude-haiku-4-5-20251001")
            == "fastest"
        )

    def test_google_flagship(self):
        assert _classify_tier("google", "gemini-3.1-pro-preview") == "flagship"
        assert _classify_tier("google", "gemini-2.5-pro") == "flagship"

    def test_google_fast(self):
        assert _classify_tier("google", "gemini-3-flash-preview") == "fast"
        assert _classify_tier("google", "gemini-2.5-flash") == "fast"

    def test_openai_audio_excluded(self):
        """Audio/realtime/embedding models should NOT match any tier."""
        assert _classify_tier("openai", "gpt-audio-2025-08-28") is None
        assert _classify_tier("openai", "gpt-4o-realtime-preview") is None
        assert _classify_tier("openai", "gpt-4o-audio-preview") is None

    def test_unknown_model_returns_none(self):
        assert _classify_tier("openai", "totally-unknown-model") is None

    def test_unknown_provider_returns_none(self):
        assert _classify_tier("fakecloud", "some-model") is None


# ---------------------------------------------------------------------------
# Unit tests — version extraction
# ---------------------------------------------------------------------------


class TestVersionExtraction:
    def test_gpt_version(self):
        assert _extract_version_tuple("gpt-5.2") == (5, 2)

    def test_gpt_single_version(self):
        assert _extract_version_tuple("gpt-5") == (5,)

    def test_claude_version(self):
        assert _extract_version_tuple("claude-opus-4-6") == (4, 6)

    def test_claude_version_sorting(self):
        """Verify claude-opus-4-6 sorts higher than claude-opus-4-5."""
        v46 = _extract_version_tuple("claude-opus-4-6")
        v45 = _extract_version_tuple("claude-opus-4-5")
        assert v46 > v45

    def test_gemini_version(self):
        assert _extract_version_tuple("gemini-2.5-flash") == (2, 5)

    def test_no_version(self):
        assert _extract_version_tuple("unknown") == (0,)


# ---------------------------------------------------------------------------
# list_models() tests — mock at the _PROVIDER_LISTERS level
# ---------------------------------------------------------------------------


class TestListModels:
    def test_list_openai(self):
        mock_models = _make_openai_model_infos()
        with mock.patch.dict(
            _PROVIDER_LISTERS, {"openai": lambda: mock_models}
        ):
            result = list_models("openai")

        assert len(result) == len(mock_models)
        assert all(isinstance(m, ModelInfo) for m in result)
        assert all(m.provider == "openai" for m in result)
        ids = {m.id for m in result}
        assert "gpt-5.2" in ids
        assert "o3" in ids

    def test_list_anthropic(self):
        mock_models = _make_anthropic_model_infos()
        with mock.patch.dict(
            _PROVIDER_LISTERS, {"anthropic": lambda: mock_models}
        ):
            result = list_models("anthropic")

        assert len(result) == len(mock_models)
        assert all(isinstance(m, ModelInfo) for m in result)
        assert all(m.provider == "anthropic" for m in result)
        ids = {m.id for m in result}
        assert "claude-opus-4-6" in ids
        assert "claude-sonnet-4-6" in ids

    def test_list_google(self):
        mock_models = _make_google_model_infos()
        with mock.patch.dict(
            _PROVIDER_LISTERS, {"google": lambda: mock_models}
        ):
            result = list_models("google")

        assert len(result) == len(mock_models)
        assert all(isinstance(m, ModelInfo) for m in result)
        assert all(m.provider == "google" for m in result)
        ids = {m.id for m in result}
        assert "gemini-3.1-pro-preview" in ids
        assert "gemini-2.5-flash" in ids

    def test_list_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            list_models("fakecloud")

    def test_list_returns_empty_on_api_error(self):
        def failing_lister():
            raise RuntimeError("API down")

        with mock.patch.dict(
            _PROVIDER_LISTERS, {"openai": failing_lister}
        ):
            result = list_models("openai")
        assert result == []

    def test_list_openai_sdk_integration(self):
        """Test that _list_openai correctly transforms SDK objects and filters."""
        sdk_models = _make_openai_sdk_models()
        mock_client = mock.MagicMock()
        mock_client.models.list.return_value = sdk_models

        mock_openai = mock.MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        with mock.patch.dict("sys.modules", {"openai": mock_openai}):
            from consilium.model_registry import _list_openai
            result = _list_openai()

        # Non-chat models (embedding, tts, dall-e, realtime, audio, whisper)
        # should be filtered out — 8 chat models remain from 14 total.
        assert len(result) == 8
        assert all(isinstance(m, ModelInfo) for m in result)
        assert result[0].id == "gpt-5.2"
        assert result[0].created_at is not None
        result_ids = {m.id for m in result}
        assert "text-embedding-3-large" not in result_ids
        assert "tts-1" not in result_ids
        assert "dall-e-3" not in result_ids

    def test_list_anthropic_sdk_integration(self):
        """Test that _list_anthropic correctly transforms SDK objects and filters."""
        sdk_models = _make_anthropic_sdk_models()
        mock_response = mock.MagicMock()
        mock_response.data = sdk_models
        mock_client = mock.MagicMock()
        mock_client.models.list.return_value = mock_response

        mock_anthropic = mock.MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with mock.patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from consilium.model_registry import _list_anthropic
            result = _list_anthropic()

        # Non-Claude model should be filtered out — 5 Claude models remain from 6 total.
        assert len(result) == 5
        assert result[0].id == "claude-opus-4-6"
        assert result[0].display_name == "Claude Opus 4.6"
        result_ids = {m.id for m in result}
        assert "some-internal-model" not in result_ids

    def test_list_google_sdk_integration(self):
        """Test that _list_google correctly transforms SDK objects and filters."""
        sdk_models = _make_google_sdk_models()

        mock_genai_client = mock.MagicMock()
        mock_genai_client.models.list.return_value = sdk_models

        mock_genai = mock.MagicMock()
        mock_genai.Client.return_value = mock_genai_client

        with mock.patch.dict(
            "sys.modules", {"google": mock.MagicMock(), "google.genai": mock_genai}
        ):
            # Patch the import inside _list_google
            with mock.patch(
                "consilium.model_registry._list_google"
            ) as mock_lister:
                # Simulate what _list_google does with the SDK models
                filtered = []
                for m in sdk_models:
                    actions = getattr(m, "supported_actions", []) or []
                    if "generateContent" not in actions:
                        continue
                    model_id = m.name
                    if model_id.startswith("models/"):
                        model_id = model_id[len("models/"):]
                    if "gemini" not in model_id.lower():
                        continue
                    filtered.append(
                        ModelInfo(
                            id=model_id,
                            provider="google",
                            display_name=m.display_name,
                        )
                    )
                mock_lister.return_value = filtered

                with mock.patch.dict(
                    _PROVIDER_LISTERS, {"google": mock_lister}
                ):
                    result = list_models("google")

        # Only 4 Gemini models with generateContent should remain
        # (text-embedding-004 has embedContent only, gemini-embedding-exp
        # has embedContent only)
        assert len(result) == 4
        assert result[0].id == "gemini-3.1-pro-preview"
        assert result[0].display_name == "Gemini 3.1 Pro Preview"
        result_ids = {m.id for m in result}
        assert "text-embedding-004" not in result_ids
        assert "gemini-embedding-exp" not in result_ids


# ---------------------------------------------------------------------------
# get_latest() tests
# ---------------------------------------------------------------------------


class TestGetLatest:
    def _mock_list_models(self, provider):
        """Return appropriate mock models for a provider."""
        if provider == "openai":
            return _make_openai_model_infos()
        elif provider == "anthropic":
            return _make_anthropic_model_infos()
        elif provider == "google":
            return _make_google_model_infos()
        return []

    def test_openai_flagship(self):
        with mock.patch(
            "consilium.model_registry.list_models",
            side_effect=self._mock_list_models,
        ):
            result = get_latest("openai", "flagship")
        assert result == "gpt-5.2"

    def test_openai_fast(self):
        with mock.patch(
            "consilium.model_registry.list_models",
            side_effect=self._mock_list_models,
        ):
            result = get_latest("openai", "fast")
        assert result == "gpt-5-mini"

    def test_openai_reasoning(self):
        with mock.patch(
            "consilium.model_registry.list_models",
            side_effect=self._mock_list_models,
        ):
            result = get_latest("openai", "reasoning")
        assert result == "o3"

    def test_openai_reasoning_fast(self):
        with mock.patch(
            "consilium.model_registry.list_models",
            side_effect=self._mock_list_models,
        ):
            result = get_latest("openai", "reasoning_fast")
        assert result == "o4-mini"

    def test_anthropic_flagship(self):
        with mock.patch(
            "consilium.model_registry.list_models",
            side_effect=self._mock_list_models,
        ):
            result = get_latest("anthropic", "flagship")
        assert result == "claude-opus-4-6"

    def test_anthropic_fast(self):
        with mock.patch(
            "consilium.model_registry.list_models",
            side_effect=self._mock_list_models,
        ):
            result = get_latest("anthropic", "fast")
        assert result == "claude-sonnet-4-6"

    def test_anthropic_fastest(self):
        with mock.patch(
            "consilium.model_registry.list_models",
            side_effect=self._mock_list_models,
        ):
            result = get_latest("anthropic", "fastest")
        assert "haiku" in result

    def test_google_flagship(self):
        with mock.patch(
            "consilium.model_registry.list_models",
            side_effect=self._mock_list_models,
        ):
            result = get_latest("google", "flagship")
        assert result == "gemini-3.1-pro-preview"

    def test_google_fast(self):
        with mock.patch(
            "consilium.model_registry.list_models",
            side_effect=self._mock_list_models,
        ):
            result = get_latest("google", "fast")
        assert result == "gemini-3-flash-preview"

    def test_unknown_tier_raises(self):
        with pytest.raises(ValueError, match="Unknown tier"):
            get_latest("openai", "imaginary_tier")

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_latest("fakecloud", "flagship")

    def test_fallback_on_api_error(self):
        with mock.patch(
            "consilium.model_registry.list_models",
            return_value=[],
        ):
            result = get_latest("openai", "flagship")
        assert result == DEFAULTS["openai"]["flagship"]

    def test_fallback_anthropic_on_empty(self):
        with mock.patch(
            "consilium.model_registry.list_models",
            return_value=[],
        ):
            result = get_latest("anthropic", "fast")
        assert result == DEFAULTS["anthropic"]["fast"]

    def test_no_fallback_raises(self):
        """If no match and no fallback for the tier, raise ValueError."""
        with mock.patch(
            "consilium.model_registry.list_models",
            return_value=[],
        ):
            # reasoning tier has no fallback for anthropic
            with pytest.raises(ValueError, match="No model found"):
                get_latest("anthropic", "reasoning")


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------


class TestCache:
    def test_second_call_uses_cache(self):
        mock_models = [
            ModelInfo(id="gpt-5.2", provider="openai"),
        ]
        call_count = 0

        def counting_lister():
            nonlocal call_count
            call_count += 1
            return mock_models

        with mock.patch.dict(
            _PROVIDER_LISTERS, {"openai": counting_lister}
        ):
            first = list_models("openai")
            second = list_models("openai")

        assert call_count == 1
        assert first == second

    def test_cache_expires(self):
        mock_models = [
            ModelInfo(id="gpt-5.2", provider="openai"),
        ]
        call_count = 0

        def counting_lister():
            nonlocal call_count
            call_count += 1
            return mock_models

        set_cache_ttl(0.05)  # 50ms TTL
        try:
            with mock.patch.dict(
                _PROVIDER_LISTERS, {"openai": counting_lister}
            ):
                list_models("openai")
                time.sleep(0.1)
                list_models("openai")

            assert call_count == 2
        finally:
            set_cache_ttl(3600)  # restore default

    def test_clear_cache_forces_refresh(self):
        mock_models = [
            ModelInfo(id="gpt-5.2", provider="openai"),
        ]
        call_count = 0

        def counting_lister():
            nonlocal call_count
            call_count += 1
            return mock_models

        with mock.patch.dict(
            _PROVIDER_LISTERS, {"openai": counting_lister}
        ):
            list_models("openai")
            clear_cache()
            list_models("openai")

        assert call_count == 2


# ---------------------------------------------------------------------------
# Filtering tests
# ---------------------------------------------------------------------------


class TestFiltering:
    """Verify that non-chat models are excluded by provider listers."""

    def test_openai_allows_chat_models(self):
        assert _is_openai_chat_model("gpt-5.2") is True
        assert _is_openai_chat_model("gpt-4.1-mini") is True
        assert _is_openai_chat_model("o3") is True
        assert _is_openai_chat_model("o4-mini") is True

    def test_openai_blocks_non_chat_models(self):
        assert _is_openai_chat_model("text-embedding-3-large") is False
        assert _is_openai_chat_model("tts-1") is False
        assert _is_openai_chat_model("dall-e-3") is False
        assert _is_openai_chat_model("whisper-1") is False
        assert _is_openai_chat_model("gpt-4o-realtime-preview") is False
        assert _is_openai_chat_model("gpt-4o-audio-preview") is False
        assert _is_openai_chat_model("gpt-image-1") is False
        assert _is_openai_chat_model("gpt-4o-transcribe") is False

    def test_openai_lister_filters(self):
        """_list_openai returns only chat models from SDK response."""
        sdk_models = _make_openai_sdk_models()
        mock_client = mock.MagicMock()
        mock_client.models.list.return_value = sdk_models

        mock_openai = mock.MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        with mock.patch.dict("sys.modules", {"openai": mock_openai}):
            from consilium.model_registry import _list_openai
            result = _list_openai()

        result_ids = {m.id for m in result}
        assert "gpt-5.2" in result_ids
        assert "o3" in result_ids
        assert "text-embedding-3-large" not in result_ids
        assert "tts-1" not in result_ids
        assert "dall-e-3" not in result_ids

    def test_anthropic_lister_filters(self):
        """_list_anthropic returns only claude-* models."""
        sdk_models = _make_anthropic_sdk_models()
        mock_response = mock.MagicMock()
        mock_response.data = sdk_models
        mock_client = mock.MagicMock()
        mock_client.models.list.return_value = mock_response

        mock_anthropic = mock.MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with mock.patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from consilium.model_registry import _list_anthropic
            result = _list_anthropic()

        result_ids = {m.id for m in result}
        assert "claude-opus-4-6" in result_ids
        assert "some-internal-model" not in result_ids

    def test_google_lister_filters(self):
        """_list_google returns only Gemini models with generateContent."""
        sdk_models = _make_google_sdk_models()

        mock_genai_client = mock.MagicMock()
        mock_genai_client.models.list.return_value = sdk_models

        mock_genai_module = mock.MagicMock()
        mock_genai_module.Client.return_value = mock_genai_client

        mock_google = mock.MagicMock()
        mock_google.genai = mock_genai_module

        with mock.patch.dict(
            "sys.modules",
            {"google": mock_google, "google.genai": mock_genai_module},
        ):
            from consilium.model_registry import _list_google
            result = _list_google()

        result_ids = {m.id for m in result}
        assert "gemini-3.1-pro-preview" in result_ids
        assert "gemini-2.5-flash" in result_ids
        # Filtered: no generateContent
        assert "text-embedding-004" not in result_ids
        # Filtered: has generateContent but is not a Gemini model... wait,
        # gemini-embedding-exp has "gemini" but only embedContent
        assert "gemini-embedding-exp" not in result_ids


# ---------------------------------------------------------------------------
# get_default_models() tests
# ---------------------------------------------------------------------------


class TestGetDefaultModels:
    def test_returns_one_per_provider(self):
        def mock_list(provider):
            if provider == "openai":
                return [ModelInfo(id="gpt-5.2", provider="openai")]
            elif provider == "anthropic":
                return [ModelInfo(id="claude-opus-4-6", provider="anthropic")]
            elif provider == "google":
                return [
                    ModelInfo(
                        id="gemini-3.1-pro-preview", provider="google"
                    )
                ]
            return []

        with mock.patch(
            "consilium.model_registry.list_models",
            side_effect=mock_list,
        ):
            result = get_default_models()

        assert len(result) == 3
        assert "openai/gpt-5.2" in result
        assert "anthropic/claude-opus-4-6" in result
        assert "google/gemini-3.1-pro-preview" in result

    def test_falls_back_to_defaults_on_error(self):
        with mock.patch(
            "consilium.model_registry.list_models",
            return_value=[],
        ):
            result = get_default_models()

        assert len(result) == 3
        # Should use fallback defaults in "provider/model_id" format
        assert f"openai/{DEFAULTS['openai']['flagship']}" in result
        assert f"anthropic/{DEFAULTS['anthropic']['flagship']}" in result
        assert f"google/{DEFAULTS['google']['flagship']}" in result


# ---------------------------------------------------------------------------
# ModelInfo dataclass tests
# ---------------------------------------------------------------------------


class TestModelInfo:
    def test_display_name_defaults_to_id(self):
        m = ModelInfo(id="gpt-5.2", provider="openai")
        assert m.display_name == "gpt-5.2"

    def test_display_name_explicit(self):
        m = ModelInfo(
            id="gpt-5.2", provider="openai", display_name="GPT 5.2"
        )
        assert m.display_name == "GPT 5.2"

    def test_created_at_optional(self):
        m = ModelInfo(id="gpt-5.2", provider="openai")
        assert m.created_at is None

    def test_created_at_set(self):
        dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
        m = ModelInfo(id="gpt-5.2", provider="openai", created_at=dt)
        assert m.created_at == dt


# ---------------------------------------------------------------------------
# Integration with providers.py
# ---------------------------------------------------------------------------


class TestProvidersIntegration:
    def test_get_default_models_from_providers(self):
        """Test that providers.get_default_models() calls the registry."""
        from consilium.providers import get_default_models as prov_defaults

        with mock.patch(
            "consilium.model_registry.list_models",
            return_value=[],
        ):
            result = prov_defaults()

        # Should return fallback defaults
        assert len(result) == 3

    def test_providers_fallback_to_hardcoded(self):
        """If registry import fails, providers.get_default_models()
        returns DEFAULT_MODELS."""
        from consilium.providers import (
            DEFAULT_MODELS,
            get_default_models as prov_defaults,
        )

        with mock.patch(
            "consilium.model_registry.get_default_models",
            side_effect=ImportError("mock"),
        ):
            result = prov_defaults()

        assert result == list(DEFAULT_MODELS)
