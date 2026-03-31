# Consilium

**Multi-LLM council for consensus-driven AI responses.**

Consilium queries multiple LLMs in parallel, has each model review the others' responses, then synthesizes the best answer through a chairman model. Inspired by [Karpathy's llm-council](https://github.com/karpathy/llm-council).

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                    STAGE 1: QUERY                       │
│                                                         │
│   ┌─────────┐   ┌──────────┐   ┌─────────────────┐     │
│   │ GPT-5.4 │   │ Claude   │   │ Gemini 3.1 Pro  │     │
│   │         │   │ Sonnet   │   │                 │     │
│   └────┬────┘   └────┬─────┘   └────────┬────────┘     │
│        │             │                  │               │
│        ▼             ▼                  ▼               │
│   Response A    Response B         Response C           │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                  STAGE 2: REVIEW                        │
│                                                         │
│   Each model reviews all anonymized responses.          │
│   Ranks them best → worst with reasoning.               │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                STAGE 3: SYNTHESIS                       │
│                                                         │
│   Chairman model synthesizes the best answer from       │
│   all responses + reviews.                              │
│                                                         │
│              ┌──────────────────┐                       │
│              │  Final Answer    │                       │
│              └──────────────────┘                       │
└─────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install openadapt-consilium
```

Or for development:

```bash
git clone https://github.com/OpenAdaptAI/openadapt-consilium.git
cd openadapt-consilium
pip install -e ".[dev]"
```

## Quick Start

Set your API keys:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AI..."
```

### Python API

```python
from consilium import Council

council = Council()
result = council.ask("What are the key differences between REST and GraphQL?")

print(result.final_answer)
print(result.cost_summary())
```

### CLI

```bash
consilium "What are the key differences between REST and GraphQL?"
```

## API Reference

### `Council`

The main orchestrator class.

```python
from consilium import Council

council = Council(
    models=["gpt-5.4", "claude-sonnet-4-6", "gemini-3.1-pro"],
    chairman="claude-sonnet-4-6",
    max_workers=8,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `models` | `list[str]` | `["gpt-5.4", "claude-sonnet-4-6", "gemini-3.1-pro"]` | Models to query in Stage 1 |
| `chairman` | `str` | `"claude-sonnet-4-6"` | Model for Stage 3 synthesis |
| `max_workers` | `int` | `8` | Max parallel threads |

#### `council.ask()`

```python
result = council.ask(
    "Your question here",
    images=[open("screenshot.png", "rb").read()],  # optional
    budget=0.50,        # max USD spend
    system="Be concise",  # system prompt for all models
    skip_review=False,  # skip Stages 2-3
    json_schema={...},  # request JSON output
)
```

**Returns:** `CouncilResult`

| Attribute | Type | Description |
|-----------|------|-------------|
| `result.final_answer` | `str` | Synthesized best answer |
| `result.individual_responses` | `list` | Each model's Stage 1 response |
| `result.reviews` | `list` | Each model's Stage 2 review |
| `result.total_cost` | `float` | Total estimated cost in USD |
| `result.total_latency_seconds` | `float` | Wall-clock time |
| `result.cost_summary()` | `str` | Pretty-printed cost breakdown |

### Agent SDK

For AI agents and automated pipelines, use the dict-based interface:

```python
from consilium import council_query

result = council_query(
    "Analyze this screenshot and list the UI elements",
    images=[screenshot_bytes],
    models=["gpt-5.4", "claude-sonnet-4-6"],
    budget=0.25,
    skip_review=True,  # fast mode: Stage 1 only
)

print(result["final_answer"])
print(result["cost"]["total_usd"])
```

**Returns:** JSON-serializable `dict` with keys:

```json
{
  "final_answer": "...",
  "individual_responses": [
    {
      "model": "openai/gpt-5.4",
      "text": "...",
      "latency_seconds": 2.1,
      "input_tokens": 1500,
      "output_tokens": 400,
      "cost_usd": 0.007
    }
  ],
  "reviews": [...],
  "cost": {
    "breakdown": {"gpt-5.4": 0.007, "claude-sonnet-4-6": 0.012},
    "total_usd": 0.019,
    "total_input_tokens": 3000,
    "total_output_tokens": 800
  },
  "total_latency_seconds": 3.2
}
```

### CLI Reference

```
consilium "prompt" [OPTIONS]

Options:
  --models TEXT      Comma-separated model IDs (default: gpt-5.4,claude-sonnet-4-6,gemini-3.1-pro)
  --chairman TEXT    Chairman model for synthesis (default: claude-sonnet-4-6)
  --image PATH      Image file to include (repeatable)
  --budget FLOAT    Max spend in USD
  --no-review       Skip Stages 2-3 (faster, cheaper)
  --system TEXT     System prompt for all models
  --json            Output raw JSON
```

**Examples:**

```bash
# Full 3-stage pipeline
consilium "Compare Python and Rust for CLI tools"

# Fast mode (Stage 1 only)
consilium "Summarize this" --no-review

# With screenshot
consilium "What's on this screen?" --image screenshot.png

# Budget-limited
consilium "Write a haiku about AI" --budget 0.10

# JSON output for piping
consilium "List 3 colors" --json | jq '.final_answer'

# Custom models
consilium "Hello" --models gpt-5.4,gemini-3.1-pro --chairman gpt-5.4
```

## Model Support

Consilium supports any model from these providers:

| Provider | Models | Env Var |
|----------|--------|---------|
| OpenAI | `gpt-5.4`, `gpt-5.4-pro`, `gpt-5`, `gpt-5-mini`, `o3`, `o4-mini` | `OPENAI_API_KEY` |
| Anthropic | `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-haiku-4-5`, `claude-sonnet-4-5` | `ANTHROPIC_API_KEY` |
| Google | `gemini-3.1-pro`, `gemini-3-flash`, `gemini-2.5-pro`, `gemini-2.5-flash` | `GOOGLE_API_KEY` |

Use any model with the `provider/model` format:

```python
council = Council(models=["openai/gpt-5.4", "anthropic/claude-sonnet-4-6"])
```

## Budget Control

Consilium tracks costs in real-time and can halt the pipeline when a budget is exceeded:

```python
result = council.ask("Expensive question", budget=0.10)

# If Stage 1 costs > $0.10, Stages 2-3 are automatically skipped
# The best Stage 1 response is returned as the final answer
```

## Error Handling

Individual model failures don't crash the council — failed responses are marked with `[ERROR: ...]` and the remaining models continue:

```python
result = council.ask("Test")
for r in result.individual_responses:
    if r.text.startswith("[ERROR:"):
        print(f"{r.model} failed: {r.text}")
```

## Development

```bash
git clone https://github.com/OpenAdaptAI/openadapt-consilium.git
cd openadapt-consilium
pip install -e ".[dev]"
pytest
```

## License

MIT
