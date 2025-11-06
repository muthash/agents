# agents.py
import asyncio
import json
import uuid
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
import traceback

# Try to import openai, but don't require it (we have fallbacks)
try:
    import openai
    _OPENAI_AVAILABLE = True
except Exception:
    openai = None
    _OPENAI_AVAILABLE = False

# ------------------ Agent container ------------------
@dataclass
class Agent:
    name: str
    instructions: Optional[str] = None
    model: Optional[str] = None
    agents_as_tools: Optional[list] = None
    handoffs: Optional[dict] = None
    output_type: Optional[Any] = None

# Handler registry
_AGENT_HANDLERS: Dict[str, Callable[[Agent, Any], Any]] = {}

def register_agent_handler(agent: Agent, handler: Callable[[Agent, Any], Any]):
    _AGENT_HANDLERS[agent.name] = handler

# ------------------ LLM wrapper ------------------
async def llm_call(system: str, user_prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.2, max_tokens: int = 800) -> str:
    """Call the underlying LLM to produce text; returns assistant text.

    Uses openai.ChatCompletion if available; otherwise raises RuntimeError.
    The function runs sync OpenAI call in threadpool so it's async-friendly.
    """
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI package not installed or not available in this environment.")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not configured in environment.")
    # We use the openai.ChatCompletion API surface (classic). Wrap in to_thread to avoid blocking.
    def _call():
        try:
            openai.api_key = api_key
            # Compose messages
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ]
            resp = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # Classic API: resp.choices[0].message.content
            return resp.choices[0].message.content
        except Exception as e:
            # surface helpful debug
            raise RuntimeError(f"LLM call failed: {e}\n{traceback.format_exc()}")
    return await asyncio.to_thread(_call)

def _extract_json_from_text(text: str):
    """Try to find JSON object in `text` and parse it. If not found, try the whole text."""
    # Attempt first to find a JSON block
    import re
    m = re.search(r'(\{.*\}|\[.*\])', text, flags=re.S)
    candidate = m.group(0) if m else text
    # Try parsing
    try:
        return json.loads(candidate)
    except Exception:
        # Try relaxed fixes: replace single quotes -> double quotes
        try:
            fix = candidate.replace("'", '"')
            return json.loads(fix)
        except Exception:
            # As last resort, return raw text
            return text

# ------------------ Runner & trace ------------------
class RunnerResult:
    def __init__(self, output: Any = None):
        self.output = output

    def final_output_as(self, typ):
        if isinstance(self.output, typ):
            return self.output
        if isinstance(self.output, dict):
            try:
                return typ(**self.output)
            except Exception as e:
                raise RuntimeError(f"Cannot convert output to {typ}: {e}")
        raise RuntimeError(f"Result cannot be converted to {typ} (type={type(self.output)})")

class Runner:
    @staticmethod
    async def run(agent: Agent, payload: Any, timeout: Optional[float] = None) -> RunnerResult:
        if agent.name not in _AGENT_HANDLERS:
            raise RuntimeError(f"No handler registered for agent '{agent.name}'")
        handler = _AGENT_HANDLERS[agent.name]
        # handler may be async or sync
        if asyncio.iscoroutinefunction(handler):
            out = await asyncio.wait_for(handler(agent, payload), timeout=timeout)
        else:
            out = await asyncio.to_thread(lambda: handler(agent, payload))
        return RunnerResult(out)

def gen_trace_id() -> str:
    return str(uuid.uuid4())

def trace(name: str, trace_id: Optional[str] = None):
    class _TraceCtx:
        def __enter__(selfi):
            print(f"[TRACE START] {name} id={trace_id}")
        def __exit__(selfi, exc_type, exc, tb):
            print(f"[TRACE END] {name} id={trace_id}")
    return _TraceCtx()
