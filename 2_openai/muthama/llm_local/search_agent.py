# search_agent.py
from dataclasses import dataclass
from typing import List, Any
from agents import Agent, register_agent_handler, llm_call, _extract_json_from_text
from planner_agent import WebSearchItem
import asyncio

@dataclass
class SearchResultItem:
    query: str
    snippet: str
    url: str

search_agent = Agent(
    name="SearchAgent",
    instructions="Execute searches and return JSON list of {query, snippet, url}.",
    model="gpt-4o-mini",
    output_type=List[SearchResultItem],
)

def _simulated_search(payload):
    items = payload if isinstance(payload, list) else (payload.searches if hasattr(payload, "searches") else [payload])
    out = []
    for it in items:
        q = it.query if hasattr(it, "query") else str(it)
        out.append({"query": q, "snippet": f"Simulated snippet for {q}", "url": f"https://example.com/search?q={q.replace(' ', '+')}"})
    return out

async def _search_handler(agent: Agent, payload):
    # payload expected to be list[WebSearchItem] or WebSearchPlan
    serial = []
    if isinstance(payload, list):
        serial = [{"query": getattr(x, "query", str(x)), "reason": getattr(x, "reason", None)} for x in payload]
    elif hasattr(payload, "searches"):
        serial = [{"query": x.query, "reason": x.reason} for x in payload.searches]
    else:
        serial = [{"query": str(payload)}]

    system = "You are a Search agent: simulate web search results. Return JSON list of objects: {query, snippet, url}."
    user = f"Run searches for: {serial}\nReturn JSON array."

    try:
        text = await llm_call(system, user, model=agent.model)
        parsed = _extract_json_from_text(text)
        if not isinstance(parsed, list):
            parsed = _simulated_search(payload)
    except Exception:
        parsed = _simulated_search(payload)

    results = []
    for it in parsed:
        q = it.get("query") if isinstance(it, dict) else str(it)
        snippet = it.get("snippet", f"Snippet for {q}")
        url = it.get("url", f"https://example.com/search?q={q.replace(' ', '+')}")
        results.append(SearchResultItem(query=q, snippet=snippet, url=url))
    return results

register_agent_handler(search_agent, _search_handler)
