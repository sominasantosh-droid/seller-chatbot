import os
import json
import time
from datetime import datetime, timedelta
from typing import AsyncGenerator, Optional

import anthropic
import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ─────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
JWT_SECRET        = os.getenv("JWT_SECRET", "change-me-in-production-secret-key")
ADMIN_USERNAME    = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD    = os.getenv("ADMIN_PASSWORD", "changeme")
MCP_SERVER_URL    = os.getenv("MCP_SERVER_URL", "http://localhost:3000/sse")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

app = FastAPI(title="Seller Chatbot API")
security = HTTPBearer()

# In-memory conversation history: { username: [{"role": ..., "content": ...}, ...] }
conversation_store: dict[str, list] = {}

SYSTEM_PROMPT = """You are an expert Amazon seller assistant integrated with the Amazon Selling Partner API (SP-API).
You help sellers manage their inventory, listings, orders, pricing, and analytics through natural conversation.

When using tools:
- Briefly explain what you're looking up and why
- Summarize results in a clear, actionable way
- Highlight any issues or opportunities you notice

Always be concise, helpful, and specific with numbers and data."""

SYSTEM_PROMPT_FALLBACK = SYSTEM_PROMPT + """

NOTE: The SP-API MCP server is currently unavailable. Answer from your training knowledge and
advise the user to ensure their MCP server is running at the configured URL."""


# ── PYDANTIC MODELS ─────────────────────────────────────────────────────
class AuthRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    message: str


# ── AUTH ────────────────────────────────────────────────────────────────
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Validate JWT and return username (sub claim)."""
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=["HS256"])
        username: str = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


# ── SSE HELPERS ─────────────────────────────────────────────────────────
def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"

def step_event(
    num: int,
    icon: str,
    step_type: str,
    title: str,
    subtitle: str = "",
    code: Optional[str] = None,
    duration: str = "",
) -> str:
    return sse({
        "type":     "step",
        "num":      num,
        "icon":     icon,
        "stepType": step_type,
        "title":    title,
        "subtitle": subtitle,
        "code":     code,
        "duration": duration,
    })

def message_chunk_event(content: str) -> str:
    return sse({"type": "message_chunk", "content": content})

def outcome_event(status: str, title: str, body: str, chip: str) -> str:
    return sse({"type": "outcome", "status": status, "title": title, "body": body, "chip": chip})

def done_event() -> str:
    return sse({"type": "done"})

def error_event(message: str) -> str:
    return sse({"type": "error", "message": message})


# ── MCP AVAILABILITY CHECK ──────────────────────────────────────────────
async def check_mcp_available() -> bool:
    """Ping the MCP server with a short timeout."""
    try:
        # Try the base URL (without /sse path) as a health probe
        base_url = MCP_SERVER_URL.replace("/sse", "")
        async with httpx.AsyncClient(timeout=3.0) as http:
            r = await http.get(base_url)
            return r.status_code < 500
    except Exception:
        return False


# ── TOOL CALL HELPERS ───────────────────────────────────────────────────
def format_tool_input(tool_name: str, tool_input: dict) -> str:
    """Format tool call as a readable code string."""
    if not tool_input:
        return f"{tool_name}()"
    pairs = ", ".join(f"{k}={repr(v)}" for k, v in list(tool_input.items())[:3])
    suffix = ", ..." if len(tool_input) > 3 else ""
    return f"{tool_name}({pairs}{suffix})"

def extract_tool_result_content(event) -> str:
    """Extract text content from a tool result event."""
    content = getattr(event, "content", None)
    if content is None:
        return "No content returned"
    if isinstance(content, str):
        return content[:300]
    if isinstance(content, list):
        parts = []
        for block in content:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
        return " ".join(parts)[:300]
    return str(content)[:300]

def resolve_tool_name(event, tool_call_map: dict) -> str:
    """Match tool_use_id back to the tool name."""
    tool_use_id = getattr(event, "tool_use_id", None)
    if tool_use_id:
        for tc in tool_call_map.values():
            if tc.get("id") == tool_use_id:
                return tc["name"]
    return "unknown_tool"


# ── CORE CHAT STREAM ─────────────────────────────────────────────────────
async def chat_stream(message: str, username: str) -> AsyncGenerator[str, None]:
    """
    Main SSE generator. Streams step cards, message chunks, and outcome cards
    as Claude processes the request with MCP tool calls.
    """
    # Retrieve or initialize conversation history
    history = conversation_store.setdefault(username, [])
    history.append({"role": "user", "content": message})

    step_counter = 0
    tool_call_map: dict[int, dict] = {}  # content_block index → {id, name, input_buf}
    bot_reply_buf = ""
    tool_results: list[tuple[str, bool, str]] = []  # (name, success, summary)
    text_step_emitted = False
    t_start = time.monotonic()

    # Step 1: Analyzing
    step_counter += 1
    yield step_event(
        num=step_counter,
        icon="🧠",
        step_type="thinking",
        title="Analyzing request",
        subtitle=f"Triggered by: User message\n→ Task: {message[:100]}",
        duration="0.2s",
    )

    # Check MCP availability
    mcp_available = await check_mcp_available()
    mcp_servers = []

    if mcp_available:
        mcp_servers = [{"type": "url", "url": MCP_SERVER_URL, "name": "sp-api"}]
        step_counter += 1
        yield step_event(
            num=step_counter,
            icon="🔐",
            step_type="auth",
            title="SP-API MCP server connected",
            subtitle=f"Connected to: {MCP_SERVER_URL}",
            code="✓ MCP server ready",
        )
    else:
        step_counter += 1
        yield step_event(
            num=step_counter,
            icon="⚠️",
            step_type="auth",
            title="MCP server unavailable — fallback mode",
            subtitle=f"Could not reach: {MCP_SERVER_URL}\n→ Claude will answer from training knowledge",
        )

    system = SYSTEM_PROMPT if mcp_available else SYSTEM_PROMPT_FALLBACK

    try:
        # Build kwargs — only pass betas+mcp_servers when MCP is available
        stream_kwargs = dict(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=system,
            messages=history,
        )
        if mcp_available:
            stream_kwargs["betas"] = ["mcp-client-0-1"]
            stream_kwargs["mcp_servers"] = mcp_servers

        t_tool = time.monotonic()

        with client.beta.messages.stream(**stream_kwargs) as stream:
            for event in stream:
                etype = getattr(event, "type", None)

                # ── Text / tool_use block starts ──────────────────────
                if etype == "content_block_start":
                    cb = event.content_block
                    cb_type = getattr(cb, "type", None)

                    if cb_type == "text" and not text_step_emitted:
                        text_step_emitted = True
                        step_counter += 1
                        yield step_event(
                            num=step_counter,
                            icon="💬",
                            step_type="thinking",
                            title="Generating response",
                            subtitle="Claude is composing the reply",
                        )

                    elif cb_type == "tool_use":
                        tool_call_map[event.index] = {
                            "id":        getattr(cb, "id", ""),
                            "name":      getattr(cb, "name", "tool"),
                            "input_buf": "",
                        }

                # ── Streaming deltas ──────────────────────────────────
                elif etype == "content_block_delta":
                    delta = event.delta
                    delta_type = getattr(delta, "type", None)

                    if delta_type == "text_delta":
                        chunk = delta.text
                        bot_reply_buf += chunk
                        yield message_chunk_event(chunk)

                    elif delta_type == "input_json_delta":
                        idx = getattr(event, "index", -1)
                        if idx in tool_call_map:
                            tool_call_map[idx]["input_buf"] += getattr(delta, "partial_json", "")

                # ── Block stopped ─────────────────────────────────────
                elif etype == "content_block_stop":
                    idx = getattr(event, "index", -1)
                    if idx in tool_call_map:
                        tc = tool_call_map[idx]
                        elapsed = f"{time.monotonic() - t_tool:.1f}s"
                        try:
                            tool_input = json.loads(tc["input_buf"]) if tc["input_buf"] else {}
                        except json.JSONDecodeError:
                            tool_input = {}

                        step_counter += 1
                        yield step_event(
                            num=step_counter,
                            icon="📡",
                            step_type="api",
                            title=f"Calling tool: {tc['name']}",
                            subtitle="SP-API MCP tool invocation",
                            code=format_tool_input(tc["name"], tool_input),
                            duration=elapsed,
                        )
                        t_tool = time.monotonic()

                # ── Tool results ──────────────────────────────────────
                elif etype == "tool_result":
                    elapsed = f"{time.monotonic() - t_tool:.1f}s"
                    is_error = getattr(event, "is_error", False)
                    result_text = extract_tool_result_content(event)
                    tool_name = resolve_tool_name(event, tool_call_map)

                    tool_results.append((tool_name, not is_error, result_text))

                    step_counter += 1
                    if not is_error:
                        yield step_event(
                            num=step_counter,
                            icon="✅",
                            step_type="success",
                            title=f"Tool result: {tool_name}",
                            subtitle=result_text[:150],
                            duration=elapsed,
                        )
                    else:
                        yield step_event(
                            num=step_counter,
                            icon="❌",
                            step_type="complete",
                            title=f"Tool error: {tool_name}",
                            subtitle=result_text[:150],
                            duration=elapsed,
                        )
                    t_tool = time.monotonic()

        # ── Final step: complete ──────────────────────────────────────
        total_elapsed = f"{time.monotonic() - t_start:.1f}s"
        step_counter += 1
        tool_summary = (
            f"✓ {len(tool_results)} tool call(s) made"
            if tool_results
            else "✓ Response generated (no tool calls)"
        )
        yield step_event(
            num=step_counter,
            icon="🎉",
            step_type="complete",
            title="Operation complete",
            subtitle=tool_summary,
            duration=total_elapsed,
        )

        # ── Emit outcome cards ────────────────────────────────────────
        if tool_results:
            for tool_name, success, summary in tool_results:
                yield outcome_event(
                    status="success" if success else "error",
                    title=f"{'✅' if success else '❌'} {tool_name}",
                    body=summary or "Tool completed.",
                    chip="✓ View result" if success else "→ Check error details",
                )
        else:
            yield outcome_event(
                status="success",
                title="✅ Response Ready",
                body="Claude answered from knowledge. No SP-API tool calls were required.",
                chip="✓ Answered",
            )

        # ── Update conversation history ───────────────────────────────
        if bot_reply_buf:
            history.append({"role": "assistant", "content": bot_reply_buf})
        # Keep last 20 messages to avoid token overflow
        if len(history) > 20:
            conversation_store[username] = history[-20:]

    except anthropic.APIStatusError as exc:
        yield error_event(f"Anthropic API error ({exc.status_code}): {exc.message}")
    except anthropic.APIConnectionError:
        yield error_event("Could not connect to Anthropic API — check your API key and network")
    except Exception as exc:
        yield error_event(f"Unexpected error: {str(exc)}")

    yield done_event()


# ── ENDPOINTS ────────────────────────────────────────────────────────────
@app.get("/status")
def status():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat() + "Z"}


@app.post("/auth")
def auth(body: AuthRequest):
    if body.username != ADMIN_USERNAME or body.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    payload = {
        "sub": body.username,
        "exp": datetime.utcnow() + timedelta(hours=24),
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")
    return {"token": token}


@app.post("/chat")
async def chat(request: Request, username: str = Depends(verify_token)):
    body = await request.json()
    message = (body.get("message") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    return StreamingResponse(
        chat_stream(message, username),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":       "keep-alive",
        },
    )


@app.get("/")
def serve_index():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)
