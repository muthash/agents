# writer_agent.py
from dataclasses import dataclass
from typing import List, Any
from agents import Agent, register_agent_handler, llm_call, _extract_json_from_text
from search_agent import SearchResultItem
import asyncio

@dataclass
class ReportData:
    title: str
    markdown_report: str
    metadata: dict = None

writer_agent = Agent(
    name="WriterAgent",
    instructions="Compose a markdown report from search results. Output JSON {title, markdown_report, metadata}.",
    model="gpt-4o-mini",
    output_type=ReportData,
)

def _simulated_writer(payload):
    # create a simple markdown
    results = payload if isinstance(payload, list) else payload.get("search_results", [])
    lines = ["# Research Report\n"]
    for i, r in enumerate(results, start=1):
        q = getattr(r, "query", r.get("query", str(r)))
        snippet = getattr(r, "snippet", r.get("snippet", ""))
        url = getattr(r, "url", r.get("url", ""))
        lines.extend([f"## Result {i}: {q}", snippet, f"[source]({url})", ""])
    md = "\n".join(lines)
    return {"title": "Research Report", "markdown_report": md, "metadata": {"count": len(results)}}

async def _writer_handler(agent: Agent, payload):
    # payload expected: list[SearchResultItem] or {"search_results": ...}
    results = payload if isinstance(payload, list) else payload.get("search_results", payload)
    # send to LLM to compose
    system = "You are a Writer agent: produce JSON {title, markdown_report, metadata} from search results."
    user = f"Search results: { [ {'query': getattr(r,'query',None), 'snippet': getattr(r,'snippet',None), 'url': getattr(r,'url',None)} for r in results ] }\nReturn one JSON object."

    try:
        text = await llm_call(system, user, model=agent.model)
        parsed = _extract_json_from_text(text)
        if not isinstance(parsed, dict):
            parsed = _simulated_writer(payload)
    except Exception:
        parsed = _simulated_writer(payload)

    return ReportData(
        title=parsed.get("title", "Research Report"),
        markdown_report=parsed.get("markdown_report", str(parsed)),
        metadata=parsed.get("metadata", {})
    )

register_agent_handler(writer_agent, _writer_handler)
