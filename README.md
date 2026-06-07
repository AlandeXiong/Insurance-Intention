# Multi-Turn Dialogue Intent Recognition System

An industrial multi-turn intent recognition system for **insurance smart marketing and customer service**. It uses **LLM dynamic intent capture + an insurance industry reference taxonomy**, with coreference resolution, intent drift detection, structured clarification guidance, and clarification reply refinement.

## Key Features

- **Dynamic intent capture**: DeepSeek / Alibaba Cloud Qwen LLMs understand user intent in natural language — not fixed enum labels
- **Insurance reference framework**: 12 common customer intent categories for LLM guidance (product inquiry, premium quote, claims service, etc.)
- **Multi-turn context management**: Coreference resolution, cross-turn slot inheritance, topic stack tracking
- **Clarification loop**: vague input → structured follow-up → user reply → intent refinement
- **Intent drift detection**: Detect topic shifts; distinguish in-chain related intent switches
- **Dual entry points**: Interactive CLI + FastAPI production API

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure LLM API (copy and edit)
cp .env.example .env
# DeepSeek: LLM_PROVIDER=deepseek + DEEPSEEK_API_KEY
# Qwen:     LLM_PROVIDER=qwen + DASHSCOPE_API_KEY

# Interactive multi-turn chat
python chat.py

# Preset scenario demo
python main.py --mode demo

# Automated tests
python tests/run_tests.py

# Start API server
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

## Project Structure

```
intention/
├── config/settings.py          # System config (latency budget, LLM, clarification thresholds)
├── src/
│   ├── pipeline.py             # Main pipeline orchestration
│   ├── domain/                 # Insurance domain reference framework
│   ├── engines/                # LLM engine + entity extraction
│   ├── context/                # Context management and coreference resolution
│   ├── clarification/          # Clarification guidance + reply refinement
│   ├── drift/                  # Intent drift detection
│   └── models/                 # Data models
├── api/server.py               # FastAPI endpoints
├── chat.py                     # Interactive chat entry point
├── main.py                     # Demo entry point
├── tests/                      # Automated tests
└── docs/
    └── ARCHITECTURE.md         # Architecture and technical documentation
```

## Documentation

| Document | Description |
|----------|-------------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Overall architecture, technical approach, implementation details |
| [.env.example](.env.example) | Environment variable reference |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider: `deepseek` / `qwen` | `deepseek` |
| `DEEPSEEK_API_KEY` | DeepSeek API key (when provider=deepseek) | — |
| `DEEPSEEK_MODEL` | DeepSeek model name | `deepseek-chat` |
| `DASHSCOPE_API_KEY` | Alibaba DashScope key (when provider=qwen) | — |
| `QWEN_MODEL` | Qwen model name | `qwen-plus` |
| `QWEN_API_BASE` | Qwen OpenAI-compatible endpoint | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `LLM_TIMEOUT_S` | Request timeout (seconds) | `30` |
| `CLARIFICATION_CONFIDENCE_THRESHOLD` | Clarification trigger confidence threshold | `0.72` |

### Switching to Qwen

```bash
# .env
LLM_PROVIDER=qwen
DASHSCOPE_API_KEY=sk-your-dashscope-key
QWEN_MODEL=qwen-plus

# Connectivity test
python scripts/test_llm.py
```

## API Examples

```bash
# Intent recognition
curl -X POST http://localhost:8000/v1/intent/predict/sync \
  -H "Content-Type: application/json" \
  -d '{"utterance": "How long is its waiting period?", "session_id": "user-001"}'

# List reference categories
curl http://localhost:8000/v1/intent/categories
```

## Target Metrics (Design Goals)

| Metric | Target |
|--------|--------|
| Intent recognition accuracy | ≥ 95% |
| Intent drift detection rate | ≥ 92% |
| Multi-intent recognition accuracy | ≥ 88% |
| End-to-end latency (incl. LLM) | ≤ 600ms (lightweight path) / 1–3s (LLM path) |

## Tech Stack

Python 3.10+ · Pydantic · httpx · FastAPI · DeepSeek / Alibaba Cloud Qwen API (OpenAI-compatible)
