# Amazon SP-API Seller Chatbot

A functional multi-platform seller chatbot with a 3-column live UI:
- **Column 1** — Real-time chat with Claude
- **Column 2** — Live agent operation steps (MCP tool calls)
- **Column 3** — Formatted API outcomes

## Prerequisites

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com)
- Your SP-API MCP server running at `http://localhost:3000/sse`

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env and set your ANTHROPIC_API_KEY
```

## Run

```bash
uvicorn app:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

The UI auto-authenticates on load (credentials: `admin` / `changeme`).

## Architecture

```
Browser (index.html)
    │  POST /auth     → JWT token
    │  POST /chat     → SSE stream
    ▼
FastAPI (app.py)
    │  client.beta.messages.stream(mcp_servers=[...])
    ▼
Anthropic Claude API
    │  tool_use events
    ▼
SP-API MCP Server (localhost:3000)
    └─ Amazon SP-API calls
```

### SSE Event Types

| Type | Description |
|---|---|
| `step` | Agent operation card for Column 2 |
| `message_chunk` | Streaming text for Column 1 bot bubble |
| `outcome` | Result card for Column 3 |
| `error` | Error message |
| `done` | Stream complete |

## Example Queries

- "Show my recent orders"
- "Get my inventory levels"
- "List products with low stock"
- "What are my top selling ASINs?"
- "Check the status of order 123-456"

## Adding Shopify MCP

Add a second MCP server to `MCP_SERVER_URL` logic in `app.py`:

```python
mcp_servers = [
    {"type": "url", "url": "http://localhost:3000/sse", "name": "sp-api"},
    {"type": "url", "url": "http://localhost:3001/sse", "name": "shopify"},
]
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | Your Anthropic API key | (required) |
| `JWT_SECRET` | Secret for signing JWTs | `change-me-...` |
| `ADMIN_USERNAME` | Login username | `admin` |
| `ADMIN_PASSWORD` | Login password | `changeme` |
| `MCP_SERVER_URL` | SP-API MCP server SSE URL | `http://localhost:3000/sse` |
