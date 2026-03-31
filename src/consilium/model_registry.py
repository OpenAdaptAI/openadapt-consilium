"""Model auto-detection registry.

Queries provider APIs for available models and classifies them by tier
(flagship, fast, reasoning, fastest). Results are TTL-cached (1 hour by
default) with hardcoded fallback defaults when API calls fail.

Usage::

    from consilium.model_registry import list_models, get_latest

    models = list_models("openai")
    best = get_latest("anthropic", tier="flagship")
"""

from __future__ import annotations

import dataclasses
import logging
import re
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ModelInfo:
    """Metadata for a single model returned by a provider API."""

    id: str
    provider: str
    created_at: datetime | None = None
    display_name: str = ""

    def __post_init__(self) -> None:
        if not self.display_name:
            self.display_name = self.id


# ---------------------------------------------------------------------------
# Hardcoded fallback defaults
# ---------------------------------------------------------------------------

DEFAULTS: Dict[str, Dict[str, str]] = {
    "openai": {
        "flagship": "gpt-5.4",
        "fast": "gpt-5-mini",
        "reasoning": "o3",
        "reasoning_fast": "o4-mini",
    },
    "anthropic": {
        "flagship": "claude-opus-4-6",
        "fast": "claude-sonnet-4-6",
        "fastest": "claude-haiku-4-5",
    },
    "google": {
        "flagship": "gemini-3.1-pro-preview",
        "fast": "gemini-3-flash-preview",
        "fastest": "gemini-2.5-flash",
    },
}

ALL_TIERS = {"flagship", "fast", "reasoning", "reasoning_fast", "fastest"}

# ---------------------------------------------------------------------------
# Tier classification regexes
# ---------------------------------------------------------------------------

# Each entry is (compiled_regex, tier_name). First match wins.
_OPENAI_TIER_PATTERNS = [
    (re.compile(r"^o\d+-mini"), "reasoning_fast"),
    (re.compile(r"^o\d+"), "reasoning"),
    (re.compile(r"^gpt-\d+(\.\d+)?-mini"), "fast"),
    (re.compile(r"^gpt-\d+(\.\d+)?-nano"), "fastest"),
    (re.compile(r"^gpt-\d+(\.\d+)?$"), "flagship"),
    (re.compile(r"^gpt-\d+(\.\d+)?-pro"), "flagship"),
]

_ANTHROPIC_TIER_PATTERNS = [
    (re.compile(r"haiku", re.IGNORECASE), "fastest"),
    (re.compile(r"sonnet", re.IGNORECASE), "fast"),
    (re.compile(r"opus", re.IGNORECASE), "flagship"),
]

_GOOGLE_TIER_PATTERNS = [
    (re.compile(r"flash-lite", re.IGNORECASE), "fastest"),
    (re.compile(r"flash", re.IGNORECASE), "fast"),
    (re.compile(r"pro", re.IGNORECASE), "flagship"),
]

_TIER_PATTERNS: Dict[str, list] = {
    "openai": _OPENAI_TIER_PATTERNS,
    "anthropic": _ANTHROPIC_TIER_PATTERNS,
    "google": _GOOGLE_TIER_PATTERNS,
}


def _classify_tier(provider: str, model_id: str) -> str | None:
    """Classify a model ID into a tier based on naming conventions.

    Returns the tier name, or ``None`` if no pattern matches.
    """
    patterns = _TIER_PATTERNS.get(provider, [])
    for regex, tier in patterns:
        if regex.search(model_id):
            return tier
    return None


# ---------------------------------------------------------------------------
# Version extraction for sorting
# ---------------------------------------------------------------------------

_VERSION_RE = re.compile(r"(\d+(?:[.\-]\d+)*)")


def _extract_version_tuple(model_id: str) -> tuple[int, ...]:
    """Extract a numeric version tuple from a model ID for sorting.

    Examples:
        "gpt-5.2" -> (5, 2)
        "claude-opus-4-6" -> (4, 6)
        "gemini-2.5-flash" -> (2, 5)

    Returns ``(0,)`` if no version is found.
    """
    matches = _VERSION_RE.findall(model_id)
    if not matches:
        return (0,)
    # Use the first version-like match; split on both dots and hyphens
    return tuple(int(p) for p in re.split(r"[.\-]", matches[0]))


# ---------------------------------------------------------------------------
# TTL cache
# ---------------------------------------------------------------------------

_DEFAULT_TTL_SECONDS = 3600  # 1 hour

_cache_lock = threading.Lock()
_cache: Dict[str, tuple[float, List[ModelInfo]]] = {}
_ttl_seconds: float = _DEFAULT_TTL_SECONDS


def set_cache_ttl(seconds: float) -> None:
    """Configure the TTL for the model list cache."""
    global _ttl_seconds
    _ttl_seconds = seconds


def clear_cache() -> None:
    """Clear the model list cache."""
    global _cache
    with _cache_lock:
        _cache.clear()


def _get_cached(provider: str) -> List[ModelInfo] | None:
    """Return cached model list if still valid, else None."""
    with _cache_lock:
        entry = _cache.get(provider)
        if entry is None:
            return None
        timestamp, models = entry
        if time.monotonic() - timestamp > _ttl_seconds:
            del _cache[provider]
            return None
        return models


def _set_cached(provider: str, models: List[ModelInfo]) -> None:
    """Store model list in cache."""
    with _cache_lock:
        _cache[provider] = (time.monotonic(), models)


# ---------------------------------------------------------------------------
# Provider API calls
# ---------------------------------------------------------------------------

_SUPPORTED_PROVIDERS = {"openai", "anthropic", "google"}

# ---------------------------------------------------------------------------
# OpenAI chat-model filters
# ---------------------------------------------------------------------------

_OPENAI_ALLOW_RE = re.compile(r"^(gpt-|o[1-9])")
_OPENAI_DENY_KEYWORDS = frozenset(
    {
        "audio", "realtime", "tts", "dall-e", "whisper", "moderation",
        "embedding", "search", "image", "transcribe", "codex", "instruct",
    }
)


def _is_openai_chat_model(model_id: str) -> bool:
    """Return True if *model_id* looks like an OpenAI chat/reasoning model."""
    if not _OPENAI_ALLOW_RE.match(model_id):
        return False
    model_lower = model_id.lower()
    return not any(kw in model_lower for kw in _OPENAI_DENY_KEYWORDS)


def _list_openai() -> List[ModelInfo]:
    """List chat-capable models from OpenAI API."""
    import openai

    client = openai.OpenAI()
    response = client.models.list()
    models = []
    for m in response:
        if not _is_openai_chat_model(m.id):
            continue
        created_at = None
        if hasattr(m, "created") and m.created:
            created_at = datetime.fromtimestamp(m.created, tz=timezone.utc)
        display_name = getattr(m, "id", str(m))
        models.append(
            ModelInfo(
                id=m.id,
                provider="openai",
                created_at=created_at,
                display_name=display_name,
            )
        )
    return models


def _list_anthropic() -> List[ModelInfo]:
    """List Claude models from Anthropic API."""
    import anthropic

    client = anthropic.Anthropic()
    response = client.models.list()
    models = []
    for m in response.data:
        model_id = m.id
        if not model_id.startswith("claude-"):
            continue
        created_at = None
        if hasattr(m, "created_at") and m.created_at:
            if isinstance(m.created_at, str):
                try:
                    created_at = datetime.fromisoformat(m.created_at)
                except (ValueError, TypeError):
                    pass
            elif isinstance(m.created_at, datetime):
                created_at = m.created_at
        display_name = getattr(m, "display_name", None) or model_id
        models.append(
            ModelInfo(
                id=model_id,
                provider="anthropic",
                created_at=created_at,
                display_name=display_name,
            )
        )
    return models


def _list_google() -> List[ModelInfo]:
    """List Gemini content-generation models from Google API."""
    from google import genai

    client = genai.Client()
    response = client.models.list()
    models = []
    for m in response:
        # Filter to models that support content generation
        supported = getattr(m, "supported_actions", None) or []
        if "generateContent" not in supported:
            continue
        model_id = getattr(m, "name", str(m))
        # Google model names are prefixed with "models/"
        if model_id.startswith("models/"):
            model_id = model_id[len("models/"):]
        # Only include Gemini models
        if "gemini" not in model_id.lower():
            continue
        display_name = getattr(m, "display_name", None) or model_id
        models.append(
            ModelInfo(
                id=model_id,
                provider="google",
                created_at=None,
                display_name=display_name,
            )
        )
    return models


_PROVIDER_LISTERS = {
    "openai": _list_openai,
    "anthropic": _list_anthropic,
    "google": _list_google,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_models(provider: str) -> List[ModelInfo]:
    """List available models from a provider's API.

    Results are cached for the configured TTL (default 1 hour).
    Falls back to empty list on API error (``get_latest`` handles
    the fallback to defaults).

    Args:
        provider: One of ``"openai"``, ``"anthropic"``, ``"google"``.

    Returns:
        List of :class:`ModelInfo` objects.

    Raises:
        ValueError: If the provider is not supported.
    """
    if provider not in _SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported: {sorted(_SUPPORTED_PROVIDERS)}"
        )

    cached = _get_cached(provider)
    if cached is not None:
        return cached

    lister = _PROVIDER_LISTERS[provider]
    try:
        models = lister()
    except Exception as exc:
        logger.warning(
            "Failed to list models from %s: %s. "
            "Falling back to defaults.",
            provider,
            exc,
        )
        return []

    _set_cached(provider, models)
    return models


def get_latest(provider: str, tier: str = "flagship") -> str:
    """Return the best (highest-versioned) model ID for a given tier.

    Calls :func:`list_models` internally, classifies each model by
    regex, and returns the one with the highest version number.
    Falls back to hardcoded defaults if the API is unavailable or
    no model matches the requested tier.

    Args:
        provider: One of ``"openai"``, ``"anthropic"``, ``"google"``.
        tier: One of ``"flagship"``, ``"fast"``, ``"reasoning"``,
            ``"reasoning_fast"``, ``"fastest"``.

    Returns:
        Model ID string (e.g. ``"gpt-5.2"``).

    Raises:
        ValueError: If the provider or tier is not recognized.
    """
    if provider not in _SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported: {sorted(_SUPPORTED_PROVIDERS)}"
        )
    if tier not in ALL_TIERS:
        raise ValueError(
            f"Unknown tier '{tier}'. Supported: {sorted(ALL_TIERS)}"
        )

    models = list_models(provider)

    # Filter to models matching the requested tier
    candidates = []
    for m in models:
        model_tier = _classify_tier(provider, m.id)
        if model_tier == tier:
            candidates.append(m)

    if candidates:
        # Sort by version (descending) and return the highest
        candidates.sort(
            key=lambda m: _extract_version_tuple(m.id), reverse=True
        )
        return candidates[0].id

    # Fallback to hardcoded defaults
    provider_defaults = DEFAULTS.get(provider, {})
    fallback = provider_defaults.get(tier)
    if fallback:
        logger.info(
            "No %s/%s model found via API; using fallback: %s",
            provider,
            tier,
            fallback,
        )
        return fallback

    raise ValueError(
        f"No model found for provider='{provider}', tier='{tier}' "
        f"and no fallback default is configured."
    )


def get_default_models() -> List[str]:
    """Return a list of default models (one flagship per provider).

    Each entry is in ``"provider/model_id"`` format so that
    :func:`consilium.providers.parse_model_string` can resolve the
    provider without needing an alias lookup.

    Tries :func:`get_latest` for each provider, falling back to
    hardcoded defaults on any error.

    Returns:
        List of ``"provider/model_id"`` strings.
    """
    result = []
    for provider, tiers in DEFAULTS.items():
        default_tier = "flagship"
        fallback = tiers.get(default_tier, next(iter(tiers.values())))
        try:
            model_id = get_latest(provider, tier=default_tier)
        except Exception:
            model_id = fallback
        result.append(f"{provider}/{model_id}")
    return result
