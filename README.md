# consilium

Query multiple LLMs, cross-review, and synthesize the best answer.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

---

## How It Works

consilium runs a 3-stage pipeline that produces higher-quality answers than any single model alone:

```
                    +-------------------+
                    |   Your Question   |
                    +--------+----------+
                             |
              +--------------+--------------+
              |              |              |
        +-----v----+  +-----v----+  +------v---+
        |  GPT-4o  |  |  Claude  |  |  Gemini  |   Stage 1: First Opinions
        +-----+----+  +-----+----+  +------+---+
              |              |              |
              +--------------+--------------+
                             |
              +--------------+--------------+
              |              |              |
        +-----v----+  +-----v----+  +------v---+
        |  GPT-4o  |  |  Claude  |  |  Gemini  |   Stage 2: Cross-Review
        | reviews  |  | reviews  |  | reviews  |   (anonymized responses)
        +-----+----+  +-----+----+  +------+---+
              |              |              |
              +--------------+--------------+
                             |
                    +--------v----------+
                    |     Chairman      |       Stage 3: Synthesis
                    |   (Claude S4.5)   |
                    +--------+----------+
                             |
                    +--------v----------+
                    |   Final Answer    |
                    +-------------------+
```

**Stage 1: First Opinions** -- All models receive the prompt in parallel and return independent answers.

**Stage 2: Cross-Review** -- Each model receives all responses (anonymized as "Response A", "Response B", etc.) and reviews them, noting strengths, weaknesses, and a ranking.

**Stage 3: Synthesis** -- A chairman model reads all responses and all reviews, then synthesizes the single best answer.

---

## Installation

```bash
pip install consilium
```

To include Google Gemini support:

```bash
pip install consilium[all]
```

---

## Quick Start

### Python

```python
from consilium import LLMCouncil

council = LLMCouncil()
result = council.ask("What causes the northern lights?")
print(result.final_answer)
print(result.cost_summary())
```

### CLI

```bash
consilium "What causes the northern lights?"
```

---

## CLI Reference

```
consilium PROMPT [OPTIONS]
```

| Flag | Description | Default |
|------|-------------|---------|
| `PROMPT` | The question or prompt to send (positional, required) | -- |
| `--models` | Comma-separated model identifiers | `gpt-4o,claude-sonnet-4-5,gemini-2.5-pro` |
| `--chairman` | Chairman model for Stage 3 synthesis | `claude-sonnet-4-5` |
| `--image` | Path to an image file (can be repeated) | -- |
| `--budget` | Maximum spend in USD | unlimited |
| `--no-review` | Skip Stages 2-3; show only individual responses | `false` |
| `--system` | System prompt prepended to all model calls | -- |
| `--json` | Output raw JSON instead of formatted text | `false` |

### Examples

```bash
# Basic query with defaults (GPT-4o + Claude + Gemini)
consilium "Explain quantum entanglement in plain English"

# Specific models, no cross-review
consilium "Write a haiku about rain" --models gpt-4o,claude-sonnet-4-5 --no-review

# Image analysis with budget cap
consilium "Describe this diagram" --image architecture.png --budget 0.25

# JSON output for scripting
consilium "List 3 sorting algorithms" --json | jq '.final_answer'

# Custom system prompt
consilium "Review this code" --system "You are a senior software engineer."
```

---

## Python SDK

### LLMCouncil class

Full control over the council pipeline:

```python
from consilium import LLMCouncil

council = LLMCouncil(
    models=["gpt-4o", "claude-sonnet-4-5", "gemini-2.5-pro"],
    chairman="claude-sonnet-4-5",
    max_workers=8,
)

result = council.ask(
    "Compare REST and GraphQL for a mobile app backend.",
    budget=0.50,
    system="You are a backend architect with 10 years of experience.",
)

# Access results
print(result.final_answer)
print(result.individual_responses)  # list of IndividualResponse
print(result.reviews)               # list of Review
print(result.cost_summary())        # formatted cost breakdown
print(result.total_cost)            # float, USD
```

### council_query() function

Simplified dict-based interface for agents and scripts:

```python
from consilium import council_query

result = council_query(
    "What are the SOLID principles?",
    models=["gpt-4o", "claude-sonnet-4-5"],
    budget=0.30,
)

print(result["final_answer"])
print(result["cost"]["total_usd"])
```

Returns a plain dict:

```json
{
  "final_answer": "...",
  "individual_responses": [
    {
      "model": "openai/gpt-4o",
      "text": "...",
      "latency_seconds": 1.234,
      "input_tokens": 150,
      "output_tokens": 400,
      "cost_usd": 0.004375
    }
  ],
  "reviews": [...],
  "cost": {
    "breakdown": {"gpt-4o": 0.004, "claude-sonnet-4-5": 0.005},
    "total_usd": 0.009,
    "total_input_tokens": 300,
    "total_output_tokens": 800
  },
  "total_latency_seconds": 3.456
}
```

### Image support

Pass PNG image bytes for vision-capable models:

```python
with open("screenshot.png", "rb") as f:
    image_data = f.read()

result = council.ask(
    "What application is shown in this screenshot?",
    images=[image_data],
)
```

### Budget control

Set a maximum spend in USD. If Stage 1 exceeds the budget, Stages 2-3 are skipped automatically:

```python
result = council.ask("Expensive question", budget=0.10)

if not result.reviews:
    print("Budget exceeded after Stage 1 -- reviews were skipped")
```

### Custom models

Use `provider/model` format for any model, or shorthand aliases:

```python
from consilium.providers import ProviderConfig

council = LLMCouncil(
    models=[
        "gpt-4o",                    # shorthand alias
        "openai/gpt-4.1",            # explicit provider/model
        ProviderConfig(               # full config object
            provider="anthropic",
            model="claude-opus-4",
            temperature=0.3,
            max_tokens=8192,
        ),
    ],
    chairman="claude-sonnet-4-5",
)
```

---

## Agent SDK

The `council_query()` function is designed for programmatic use in AI agent pipelines. It accepts and returns plain dicts and strings, making it straightforward to integrate:

```python
from consilium import council_query

def agent_step(observation: str) -> str:
    """Use the council to decide the next action."""
    result = council_query(
        f"Given this observation, what action should I take?\n\n{observation}",
        models=["gpt-4o", "claude-sonnet-4-5"],
        budget=0.05,
        skip_review=True,  # fast mode for real-time agents
    )
    return result["final_answer"]
```

For structured output, pass a `json_schema` parameter:

```python
result = council_query(
    "List the top 3 actions to take.",
    json_schema={"type": "object", "properties": {"actions": {"type": "array"}}},
)
```

---

## Cost Tracking

Every query tracks token usage and estimated cost:

```python
result = council.ask("What is the meaning of life?")
print(result.cost_summary())
```

Output:

```
Cost breakdown:
  claude-sonnet-4-5: $0.0123
  gemini-2.5-pro: $0.0045
  gpt-4o: $0.0089
  TOTAL: $0.0257
  Tokens: 1250 in / 890 out
```

Access programmatically:

```python
result.total_cost          # float, total USD
result.cost_breakdown      # dict: model -> USD
result.cost_tracker        # CostTracker with full detail
```

---

## Supported Models

| Alias | Provider | Model |
|-------|----------|-------|
| `gpt-4o` | openai | gpt-4o |
| `gpt-4o-mini` | openai | gpt-4o-mini |
| `gpt-4.1` | openai | gpt-4.1 |
| `gpt-4.1-mini` | openai | gpt-4.1-mini |
| `gpt-5.3` | openai | gpt-5.3 |
| `o3-mini` | openai | o3-mini |
| `claude-sonnet-4-5` | anthropic | claude-sonnet-4-5 |
| `claude-sonnet-4` | anthropic | claude-sonnet-4 |
| `claude-opus-4` | anthropic | claude-opus-4 |
| `claude-opus-4-6` | anthropic | claude-opus-4-6 |
| `gemini-2.5-pro` | google | gemini-2.5-pro |
| `gemini-3.5-pro` | google | gemini-3.5-pro |

Any model can also be specified as `provider/model` (e.g., `openai/gpt-4o`).

---

## Configuration

consilium reads API keys from environment variables:

| Variable | Provider |
|----------|----------|
| `OPENAI_API_KEY` | OpenAI (GPT-4o, o3-mini, etc.) |
| `ANTHROPIC_API_KEY` | Anthropic (Claude) |
| `GOOGLE_API_KEY` | Google (Gemini) -- requires `consilium[google]` |

You can also pass keys explicitly:

```python
from consilium.providers import ProviderConfig

config = ProviderConfig(
    provider="openai",
    model="gpt-4o",
    api_key="sk-...",
)
council = LLMCouncil(models=[config])
```

---

## Inspired By

This project is inspired by Andrej Karpathy's [llm-council](https://github.com/karpathy/llm-council), which demonstrates the value of querying multiple LLMs and having them review each other's work.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
