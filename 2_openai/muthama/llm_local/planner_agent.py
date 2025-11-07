# planner_agent.py
from dataclasses import dataclass
from typing import List, Optional
from agents import Agent, register_agent_handler, llm_call, _extract_json_from_text
import re
import asyncio

@dataclass
class WebSearchItem:
    query: str
    reason: Optional[str] = None
    priority: float = 0.5
    rank: Optional[int] = None
    tags: Optional[List[str]] = None

@dataclass
class WebSearchPlan:
    searches: List[WebSearchItem]

planner_agent = Agent(
    name="PlannerAgent",
    instructions="Produce a JSON WebSearchPlan containing searches: [{query, reason, priority, rank, tags}].",
    model="gpt-4o-mini",
    output_type=WebSearchPlan,
)

# fallback simulated logic (used if llm_call fails)
def _simulated_planner(payload: str):
    # crude splits
    parts = re.split(r',| about | for | on | regarding | and ', str(payload), flags=re.I)
    parts = [p.strip() for p in parts if p.strip()]
    out = []
    for i, p in enumerate(parts[:3], start=1):
        out.append({"query": f"{p} overview", "reason": "overview", "priority": 0.9 - i*0.1, "rank": i})
    if not out:
        out = [{"query": payload, "reason": "keyword", "priority": 0.5, "rank": 1}]
    return {"searches": out}

async def _planner_handler(agent: Agent, payload: str):
    system = "You are a Planner agent. Given a user query, produce a JSON object matching: {\"searches\": [{\"query\": str, \"reason\": str, \"priority\": float, \"rank\": int}] }."
    user = f"User query: {payload}\n\nReturn exactly one JSON object with a 'searches' array of 1..6 items."
    try:
        text = await llm_call(system, user, model=agent.model)
        parsed = _extract_json_from_text(text)
        if isinstance(parsed, str):
            # llm returned raw text unparseable -> fallback
            parsed = _simulated_planner(payload)
    except Exception:
        parsed = _simulated_planner(payload)

    # Normalize into WebSearchPlan
    searches = []
    for i, it in enumerate(parsed.get("searches", []), start=1):
        q = it.get("query") if isinstance(it, dict) else str(it)
        searches.append(WebSearchItem(
            query=q,
            reason=(it.get("reason") if isinstance(it, dict) else None),
            priority=float(it.get("priority", 0.5)) if isinstance(it, dict) else 0.5,
            rank=int(it.get("rank", i)) if isinstance(it, dict) else i,
            tags=it.get("tags") if isinstance(it, dict) else None
        ))
    return WebSearchPlan(searches=searches)

register_agent_handler(planner_agent, _planner_handler)
