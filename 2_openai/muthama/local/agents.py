# agents.py
import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# Simple Agent container
@dataclass
class Agent:
    name: str
    instructions: Optional[str] = None
    model: Optional[str] = None
    agents_as_tools: Optional[List["Agent"]] = None
    handoffs: Optional[Dict[str, Any]] = None
    output_type: Optional[Any] = None

# Registry for agent handlers (handler: async function(agent, input) -> result)
_AGENT_HANDLERS: Dict[str, Callable[[Agent, Any], Any]] = {}

def register_agent_handler(agent: Agent, handler: Callable[[Agent, Any], Any]):
    """Register an async handler for an agent instance."""
    _AGENT_HANDLERS[agent.name] = handler

class RunnerResult:
    def __init__(self, output: Any = None):
        self.output = output

    def final_output_as(self, typ):
        # Try to cast/validate naive
        if isinstance(self.output, typ):
            return self.output
        # If it's a dict and typ is a dataclass-like, try to construct
        if isinstance(self.output, dict):
            try:
                return typ(**self.output)
            except Exception as e:
                raise RuntimeError(f"Cannot convert output to {typ}: {e}")
        raise RuntimeError(f"Result cannot be converted to {typ} (type={type(self.output)})")

# A naive Runner that calls the registered handler for an agent
class Runner:
    @staticmethod
    async def run(agent: Agent, payload: Any, timeout: Optional[float] = None) -> RunnerResult:
        if agent.name not in _AGENT_HANDLERS:
            raise RuntimeError(f"No handler registered for agent '{agent.name}'")
        handler = _AGENT_HANDLERS[agent.name]
        # call handler (may be sync or async)
        if asyncio.iscoroutinefunction(handler):
            out = await asyncio.wait_for(handler(agent, payload), timeout=timeout)
        else:
            out = handler(agent, payload)
        return RunnerResult(out)

# Tracing helpers (simple prints)
def gen_trace_id() -> str:
    return str(uuid.uuid4())

def trace(name: str, trace_id: Optional[str] = None):
    # context manager stub
    class _TraceCtx:
        def __enter__(selfi):
            print(f"[TRACE START] {name} id={trace_id}")
        def __exit__(selfi, exc_type, exc, tb):
            print(f"[TRACE END] {name} id={trace_id}")
    return _TraceCtx()
