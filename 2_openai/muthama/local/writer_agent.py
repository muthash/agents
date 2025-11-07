# writer_agent.py
from dataclasses import dataclass, asdict
from typing import List, Any
from agents import Agent, register_agent_handler
from search_agent import SearchResultItem

@dataclass
class ReportData:
    title: str
    markdown_report: str
    metadata: dict = None

writer_agent = Agent(
    name="WriterAgent",
    instructions="Compose a markdown report from search results or findings.",
    model="simulated-writer-model",
    output_type=ReportData,
)

def _writer_handler(agent: Agent, payload):
    # payload may be: list[SearchResultItem] or dict with 'findings'
    findings = []
    if isinstance(payload, list) and all(hasattr(x, "snippet") for x in payload):
        findings = payload
    elif isinstance(payload, dict) and "search_results" in payload:
        findings = payload["search_results"]
    else:
        # fallback: try to stringify payload
        findings = [str(payload)]
    # Build a markdown report
    title = "Research Report"
    lines = [f"# {title}", ""]
    if isinstance(findings, list) and findings and hasattr(findings[0], "snippet"):
        for i, item in enumerate(findings, start=1):
            lines.append(f"## Result {i}: {item.query}")
            lines.append(item.snippet)
            lines.append(f"[source]({item.url})")
            lines.append("")
    else:
        lines.append("Findings:")
        lines.append(str(findings))
    md = "\n".join(lines)
    return ReportData(title=title, markdown_report=md, metadata={"count": len(findings)})

register_agent_handler(writer_agent, _writer_handler)
