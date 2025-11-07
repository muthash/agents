# search_agent.py
from dataclasses import dataclass
from typing import List, Any
from agents import Agent, register_agent_handler
from planner_agent import WebSearchItem

@dataclass
class SearchResultItem:
    query: str
    snippet: str
    url: str

search_agent = Agent(
    name="SearchAgent",
    instructions="Execute each WebSearchItem and return a list of SearchResultItem.",
    model="simulated-search-model",
    output_type=List[SearchResultItem],
)

# Very small simulated search handler
def _search_handler(agent: Agent, payload):
    # payload expected to be list[WebSearchItem] or a WebSearchPlan
    items = []
    if isinstance(payload, list):
        items = payload
    elif hasattr(payload, "searches"):
        items = payload.searches
    else:
        # if passed a raw string, create single search
        items = [WebSearchItem(query=str(payload))]
    results = []
    for it in items:
        q = it.query
        snippet = f"Simulated snippet for '{q}': key findings and summary."
        url = f"https://example.com/search?q={q.replace(' ', '+')}"
        results.append(SearchResultItem(query=q, snippet=snippet, url=url))
    return results

register_agent_handler(search_agent, _search_handler)
