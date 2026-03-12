import os
import json
import time
from contextlib import asynccontextmanager
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
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

load_dotenv()

# ── CONFIG ─────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
JWT_SECRET        = os.getenv("JWT_SECRET", "change-me-in-production-secret-key")
ADMIN_USERNAME    = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD    = os.getenv("ADMIN_PASSWORD", "changeme")
MCP_SERVER_URL    = os.getenv("MCP_SERVER_URL", "http://localhost:3000/mcp")

TOOLS_CACHE_TTL      = 300.0  # cache tool schemas for 5 minutes
AVAILABLE_CACHE_TTL  = 30.0   # cache MCP availability check for 30 seconds

client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

# ── PERSISTENT HTTP CLIENT (created once, reused across all requests) ──
_http_client: Optional[httpx.AsyncClient] = None

# ── TOOLS CACHE ─────────────────────────────────────────────────────────
_tools_cache: list[dict] = []
_tools_cache_time: float = 0.0

# ── AVAILABILITY CACHE ──────────────────────────────────────────────────
_available_cache: Optional[bool] = None
_available_cache_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_client
    _http_client = httpx.AsyncClient(timeout=30.0, limits=httpx.Limits(max_connections=20))
    yield
    await _http_client.aclose()


app = FastAPI(title="Seller Chatbot API", lifespan=lifespan)
security = HTTPBearer()

# In-memory conversation history: { username: [{"role": ..., "content": ...}, ...] }
conversation_store: dict[str, list] = {}

SYSTEM_PROMPT = """You are an expert Amazon seller assistant integrated with the Amazon Selling Partner API (SP-API).
You help sellers manage their inventory, listings, orders, pricing, and analytics through natural conversation.

## When NOT to use tools (answer directly from knowledge):
- Greetings, small talk, or general questions ("hello", "what can you do?", "how are you?")
- Questions about your own capabilities or available tools (answer from the tool list injected below)
- General Amazon selling advice that doesn't require live data
- Questions about how SP-API works conceptually

## Tool usage rules — follow these strictly:

1. Only call tools when the user explicitly asks for LIVE data (orders, inventory, listings, pricing, reports).

2. To call any SP-API endpoint, use this two-step process:
   a. `explore-sp-api-catalog` with `action: "search"` and a short keyword (e.g. "orders", "listings", "inventory") to get the exact endpoint ID and its required parameters.
   b. `execute-sp-api` using that confirmed endpoint ID.

3. NEVER call `explore-sp-api-catalog` without a search keyword. NEVER use listCategories or list all categories/endpoints — it is slow and forbidden.
   - BAD: `explore-sp-api-catalog({listCategories: true})` or any call that lists everything
   - GOOD: `explore-sp-api-catalog({action: "search", keyword: "orders"})`

4. Only use `action: "get_endpoint"` if the search result is missing required parameter details.

5. Never guess or invent endpoint names — only use IDs confirmed by the search result.

6. When presenting results, summarize clearly with specific numbers and data."""

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
    """Ping the MCP server, caching the result for AVAILABLE_CACHE_TTL seconds."""
    global _available_cache, _available_cache_time
    now = time.monotonic()
    if _available_cache is not None and (now - _available_cache_time) < AVAILABLE_CACHE_TTL:
        return _available_cache
    try:
        base_url = MCP_SERVER_URL.replace("/mcp", "")
        r = await _http_client.get(base_url, timeout=3.0)
        result = r.status_code < 500
    except Exception:
        result = False
    _available_cache = result
    _available_cache_time = now
    return result


# ── MCP SESSION CONTEXT MANAGER ─────────────────────────────────────────
@asynccontextmanager
async def maybe_mcp_session(available: bool):
    """Open a single MCP session if the server is available, else yield None.

    IMPORTANT: We track whether we have already yielded.
    - If connection fails BEFORE yielding → yield None as fallback.
    - If an exception comes FROM the body (via athrow) AFTER yielding → re-raise it
      so it propagates correctly to chat_stream's outer except handlers.
      (Doing `yield None` after athrow() causes RuntimeError from asynccontextmanager.)
    """
    if not available:
        yield None
        return
    yielded = False
    try:
        async with streamable_http_client(MCP_SERVER_URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yielded = True
                yield session
    except BaseException as _exc:
        if not yielded:
            yield None   # Connection failed before we ever yielded — safe to yield fallback
        else:
            raise        # Exception came from the body; re-raise so caller sees the real error


# ── TOOL CALL HELPERS ───────────────────────────────────────────────────
def format_tool_input(tool_name: str, tool_input: dict) -> str:
    """Format tool call as a readable code string."""
    if not tool_input:
        return f"{tool_name}()"
    pairs = ", ".join(f"{k}={repr(v)}" for k, v in list(tool_input.items())[:3])
    suffix = ", ..." if len(tool_input) > 3 else ""
    return f"{tool_name}({pairs}{suffix})"


# ── CORE CHAT STREAM ─────────────────────────────────────────────────────
async def chat_stream(message: str, username: str) -> AsyncGenerator[str, None]:
    """
    Main SSE generator. Opens one MCP session per request and reuses it
    for all tool calls, avoiding reconnect overhead on each invocation.
    """
    history = conversation_store.setdefault(username, [])
    history.append({"role": "user", "content": message})

    step_counter = 0
    all_tool_results: list[tuple[str, bool, str]] = []
    bot_reply_buf = ""
    t_start = time.monotonic()

    step_counter += 1
    yield step_event(
        num=step_counter,
        icon="🧠",
        step_type="thinking",
        title="Analyzing request",
        subtitle=f"Triggered by: User message\n→ Task: {message[:100]}",
        duration="0.2s",
    )

    mcp_available = await check_mcp_available()
    mcp_tools: list[dict] = []
    system = SYSTEM_PROMPT if mcp_available else SYSTEM_PROMPT_FALLBACK

    try:
        async with maybe_mcp_session(mcp_available) as mcp_session:

            # ── Load tools (cached — avoids re-fetching 346 schemas per request) ─
            if mcp_session:
                global _tools_cache, _tools_cache_time
                now = time.monotonic()
                if _tools_cache and (now - _tools_cache_time) < TOOLS_CACHE_TTL:
                    mcp_tools = _tools_cache
                    cache_note = " (cached)"
                else:
                    result = await mcp_session.list_tools()
                    mcp_tools = [
                        {
                            "name": t.name,
                            "description": t.description or "",
                            "input_schema": t.inputSchema,
                        }
                        for t in result.tools
                    ]
                    _tools_cache = mcp_tools
                    _tools_cache_time = now
                    cache_note = ""
                # Inject tool count into system prompt so Claude can answer
                # meta questions ("how many endpoints?") without a tool call.
                tool_names = ", ".join(t["name"] for t in mcp_tools)
                system = system + (
                    f"\n\n## Available tools ({len(mcp_tools)} total)\n"
                    f"You have access to exactly {len(mcp_tools)} SP-API tools: {tool_names}.\n"
                    "When asked how many endpoints/tools/APIs you have, answer directly from this list — "
                    "do NOT make a tool call just to count them."
                )

                step_counter += 1
                yield step_event(
                    num=step_counter,
                    icon="🔐",
                    step_type="auth",
                    title="SP-API MCP server connected",
                    subtitle=f"Connected to: {MCP_SERVER_URL}\n→ {len(mcp_tools)} tool(s) available{cache_note}",
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

            # ── Agentic loop ──────────────────────────────────────────
            current_messages = list(history)
            MAX_ITERATIONS = 10

            for _iteration in range(MAX_ITERATIONS):
                stream_kwargs: dict = dict(
                    model="claude-sonnet-4-6",
                    max_tokens=4096,
                    system=system,
                    messages=current_messages,
                )
                if mcp_tools:
                    stream_kwargs["tools"] = mcp_tools

                text_step_emitted = False
                tool_call_map: dict[int, dict] = {}
                t_tool = time.monotonic()

                async with client.messages.stream(**stream_kwargs) as stream:
                    async for event in stream:
                        etype = getattr(event, "type", None)

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

                        elif etype == "content_block_stop":
                            idx = getattr(event, "index", -1)
                            if idx in tool_call_map:
                                tc = tool_call_map[idx]
                                elapsed = f"{time.monotonic() - t_tool:.1f}s"
                                try:
                                    tool_input = json.loads(tc["input_buf"]) if tc["input_buf"] else {}
                                except json.JSONDecodeError:
                                    tool_input = {}
                                tc["input"] = tool_input

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

                    final_msg = await stream.get_final_message()
                    stop_reason = final_msg.stop_reason
                    assistant_content = final_msg.content

                # Serialize only the API-accepted fields per block type.
                # block.model_dump() includes internal SDK fields (e.g. parsed_output)
                # that Anthropic's API rejects as "Extra inputs are not permitted".
                def _serialize_block(b) -> dict:
                    if b.type == "text":
                        return {"type": "text", "text": b.text}
                    if b.type == "tool_use":
                        return {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
                    # Fallback: strip None values and hope for the best
                    return {k: v for k, v in b.model_dump().items() if v is not None}

                current_messages.append({
                    "role": "assistant",
                    "content": [_serialize_block(b) for b in assistant_content],
                })

                if stop_reason != "tool_use":
                    break

                # ── Execute tool calls via the shared session ─────────
                tool_result_blocks = []
                for block in assistant_content:
                    if getattr(block, "type", None) != "tool_use":
                        continue
                    t_tool = time.monotonic()
                    tool_name = block.name
                    tool_input = block.input or {}
                    try:
                        if mcp_session is None:
                            raise RuntimeError("No MCP session")
                        result = await mcp_session.call_tool(tool_name, tool_input)
                        parts = []
                        for content in result.content:
                            if hasattr(content, "text"):
                                parts.append(content.text)
                            elif isinstance(content, dict) and "text" in content:
                                parts.append(content["text"])
                        result_text = "\n".join(parts) if parts else "Tool completed successfully"
                        elapsed = f"{time.monotonic() - t_tool:.1f}s"
                        all_tool_results.append((tool_name, True, result_text))
                        step_counter += 1
                        yield step_event(
                            num=step_counter,
                            icon="✅",
                            step_type="success",
                            title=f"Tool result: {tool_name}",
                            subtitle=result_text[:150],
                            duration=elapsed,
                        )
                        tool_result_blocks.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_text,
                        })
                    except Exception as e:
                        elapsed = f"{time.monotonic() - t_tool:.1f}s"
                        err_text = str(e)
                        all_tool_results.append((tool_name, False, err_text))
                        step_counter += 1
                        yield step_event(
                            num=step_counter,
                            icon="❌",
                            step_type="complete",
                            title=f"Tool error: {tool_name}",
                            subtitle=err_text[:150],
                            duration=elapsed,
                        )
                        tool_result_blocks.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Error: {err_text}",
                            "is_error": True,
                        })

                current_messages.append({"role": "user", "content": tool_result_blocks})

        # ── Final step ────────────────────────────────────────────────
        total_elapsed = f"{time.monotonic() - t_start:.1f}s"
        step_counter += 1
        tool_summary = (
            f"✓ {len(all_tool_results)} tool call(s) made"
            if all_tool_results
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

        if all_tool_results:
            for tool_name, success, summary in all_tool_results:
                yield outcome_event(
                    status="success" if success else "error",
                    title=f"{'✅' if success else '❌'} {tool_name}",
                    body=summary[:300] or "Tool completed.",
                    chip="✓ View result" if success else "→ Check error details",
                )
        else:
            yield outcome_event(
                status="success",
                title="✅ Response Ready",
                body="Claude answered from knowledge. No SP-API tool calls were required.",
                chip="✓ Answered",
            )

        if bot_reply_buf:
            history.append({"role": "assistant", "content": bot_reply_buf})
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
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "cache": {
            "tools_cached": bool(_tools_cache),
            "tools_count": len(_tools_cache),
            "tools_age_s": round(time.monotonic() - _tools_cache_time, 1) if _tools_cache else None,
            "mcp_available_cached": _available_cache,
            "mcp_available_age_s": round(time.monotonic() - _available_cache_time, 1) if _available_cache is not None else None,
        },
    }


@app.post("/cache/clear")
def cache_clear(username: str = Depends(verify_token)):
    """Force-expire all in-memory caches (tools + availability). Requires auth."""
    global _tools_cache, _tools_cache_time, _available_cache, _available_cache_time
    _tools_cache = []
    _tools_cache_time = 0.0
    _available_cache = None
    _available_cache_time = 0.0
    return {"cleared": True}


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
