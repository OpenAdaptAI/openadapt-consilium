"""LLM provider adapters.

Each provider implements a query function that accepts a text prompt and
optional images, returning ``(response_text, token_usage)``.
"""

from __future__ import annotations

import base64
import dataclasses
import os
from typing import Any, Dict, List

from consilium.cost import TokenUsage


@dataclasses.dataclass
class ProviderConfig:
    """Configuration for a single LLM provider/model."""

    provider: str  # "openai", "anthropic", "google"
    model: str  # e.g. "gpt-4.1", "claude-sonnet-4-5-20250514"
    api_key: str | None = None  # falls back to env var
    temperature: float = 0.7
    max_tokens: int = 4096

    @property
    def display_name(self) -> str:
        return f"{self.provider}/{self.model}"


def _resolve_api_key(provider: str, explicit_key: str | None) -> str:
    """Resolve API key from explicit value or env var."""
    if explicit_key:
        return explicit_key

    env_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }
    env_var = env_map.get(provider)
    if env_var:
        val = os.environ.get(env_var)
        if val:
            return val

    raise ValueError(
        f"No API key found for provider '{provider}'. "
        f"Set {env_map.get(provider, provider.upper() + '_API_KEY')} "
        f"or pass api_key= explicitly."
    )


# ---------------------------------------------------------------------------
# Provider query functions
# ---------------------------------------------------------------------------


def _query_openai(
    config: ProviderConfig,
    prompt: str,
    images: list[bytes] | None = None,
    system: str | None = None,
    json_schema: dict | None = None,
) -> tuple[str, TokenUsage]:
    """Query an OpenAI-compatible model."""
    import openai

    api_key = _resolve_api_key("openai", config.api_key)
    client = openai.OpenAI(api_key=api_key)

    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if images:
        for img_bytes in images:
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )
    messages.append({"role": "user", "content": content})

    kwargs: Dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if json_schema:
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content or ""
    usage = TokenUsage(
        model=config.model,
        input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
        output_tokens=resp.usage.completion_tokens if resp.usage else 0,
    )
    return text, usage


def _query_anthropic(
    config: ProviderConfig,
    prompt: str,
    images: list[bytes] | None = None,
    system: str | None = None,
    json_schema: dict | None = None,
) -> tuple[str, TokenUsage]:
    """Query an Anthropic model."""
    import anthropic

    api_key = _resolve_api_key("anthropic", config.api_key)
    client = anthropic.Anthropic(api_key=api_key)

    content: list[dict[str, Any]] = []
    if images:
        for img_bytes in images:
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64,
                    },
                }
            )
    content.append({"type": "text", "text": prompt})

    kwargs: Dict[str, Any] = {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "messages": [{"role": "user", "content": content}],
    }
    if system:
        kwargs["system"] = system
    if config.temperature is not None:
        kwargs["temperature"] = config.temperature

    resp = client.messages.create(**kwargs)
    text = resp.content[0].text if resp.content else ""
    usage = TokenUsage(
        model=config.model,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
    )
    return text, usage


def _query_google(
    config: ProviderConfig,
    prompt: str,
    images: list[bytes] | None = None,
    system: str | None = None,
    json_schema: dict | None = None,
) -> tuple[str, TokenUsage]:
    """Query a Google Gemini model."""
    import google.generativeai as genai

    api_key = _resolve_api_key("google", config.api_key)
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        config.model, system_instruction=system
    )

    parts: list[Any] = []
    if images:
        for img_bytes in images:
            parts.append({"mime_type": "image/png", "data": img_bytes})
    parts.append(prompt)

    resp = model.generate_content(
        parts,
        generation_config=genai.GenerationConfig(
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
        ),
    )
    text = resp.text or ""

    input_tokens = 0
    output_tokens = 0
    if hasattr(resp, "usage_metadata") and resp.usage_metadata:
        input_tokens = (
            getattr(resp.usage_metadata, "prompt_token_count", 0) or 0
        )
        output_tokens = (
            getattr(resp.usage_metadata, "candidates_token_count", 0) or 0
        )

    usage = TokenUsage(
        model=config.model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    return text, usage


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_PROVIDERS = {
    "openai": _query_openai,
    "anthropic": _query_anthropic,
    "google": _query_google,
}


def query_model(
    config: ProviderConfig,
    prompt: str,
    images: list[bytes] | None = None,
    system: str | None = None,
    json_schema: dict | None = None,
) -> tuple[str, TokenUsage]:
    """Dispatch a query to the appropriate provider.

    Args:
        config: Provider/model configuration.
        prompt: The text prompt.
        images: Optional list of image bytes (PNG).
        system: Optional system prompt.
        json_schema: If set, request JSON output (best-effort).

    Returns:
        ``(response_text, TokenUsage)``
    """
    fn = _PROVIDERS.get(config.provider)
    if fn is None:
        raise ValueError(
            f"Unknown provider '{config.provider}'. "
            f"Supported: {sorted(_PROVIDERS.keys())}"
        )
    return fn(
        config, prompt, images=images, system=system, json_schema=json_schema
    )


# ---------------------------------------------------------------------------
# Parse "provider/model" shorthand
# ---------------------------------------------------------------------------

MODEL_ALIASES: Dict[str, tuple[str, str]] = {
    # OpenAI
    "gpt-4.1": ("openai", "gpt-4.1"),
    "gpt-4.1-mini": ("openai", "gpt-4.1-mini"),
    "gpt-4.1-nano": ("openai", "gpt-4.1-nano"),
    "gpt-4o": ("openai", "gpt-4o"),
    "gpt-4o-mini": ("openai", "gpt-4o-mini"),
    "o3": ("openai", "o3"),
    "o3-mini": ("openai", "o3-mini"),
    "o4-mini": ("openai", "o4-mini"),
    # Anthropic
    "claude-haiku-4-5-20251001": ("anthropic", "claude-haiku-4-5-20251001"),
    "claude-sonnet-4-5-20250514": ("anthropic", "claude-sonnet-4-5-20250514"),
    "claude-opus-4-20250514": ("anthropic", "claude-opus-4-20250514"),
    # Google
    "gemini-2.5-flash": ("google", "gemini-2.5-flash"),
    "gemini-2.5-pro": ("google", "gemini-2.5-pro"),
}


def parse_model_string(model_str: str) -> ProviderConfig:
    """Parse a model string like ``"gpt-4.1"`` or ``"openai/gpt-4.1"``
    into a :class:`ProviderConfig`.
    """
    if "/" in model_str:
        provider, model = model_str.split("/", 1)
        return ProviderConfig(provider=provider, model=model)

    if model_str in MODEL_ALIASES:
        provider, model = MODEL_ALIASES[model_str]
        return ProviderConfig(provider=provider, model=model)

    raise ValueError(
        f"Cannot parse model '{model_str}'. Use 'provider/model' format "
        f"or one of: {sorted(MODEL_ALIASES.keys())}"
    )


# Default council composition — latest flagship from each provider
DEFAULT_MODELS: List[str] = [
    "gpt-4.1",
    "claude-sonnet-4-5-20250514",
    "gemini-2.5-pro",
]

DEFAULT_CHAIRMAN: str = "claude-sonnet-4-5-20250514"
