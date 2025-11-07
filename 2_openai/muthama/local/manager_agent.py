# manager_agent.py
from typing import Any, Dict, List
from agents import Agent, Runner, trace, gen_trace_id, register_agent_handler
from planner_agent import planner_agent, WebSearchPlan, WebSearchItem
from search_agent import search_agent, SearchResultItem
from writer_agent import writer_agent, ReportData
from email_agent import email_agent

INSTRUCTIONS = (
    "You are the Research Manager and may call the following agent tools: PlannerAgent, SearchAgent, "
    "WriterAgent, EmailAgent. Coordinate them to produce a ReportData from a user's query. "
    "You can request the planner to produce searches, then call the search agent to gather results, "
    "then call the writer to compose a report, and optionally call the email agent to send it."
)

HANDOFFS = {
    "to_planner": {"tool_name": "PlannerAgent", "expects": "str"},
    "to_search": {"tool_name": "SearchAgent", "expects": "list[WebSearchItem]"},
    "to_writer": {"tool_name": "WriterAgent", "expects": "list[SearchResultItem]"},
    "to_email": {"tool_name": "EmailAgent", "expects": "ReportData or str"},
    "done": {"tool_name": None, "expects": "ReportData"},
}

manager_agent = Agent(
    name="ManagerAgent",
    instructions=INSTRUCTIONS,
    model="simulated-manager-model",
    agents_as_tools=[planner_agent, search_agent, writer_agent, email_agent],
    handoffs=HANDOFFS,
    output_type=ReportData,
)

# Register a handler for the manager which orchestrates by calling other agents via Runner.run
async def _manager_handler(agent: Agent, payload: Any):
    query = payload if isinstance(payload, str) else str(payload.get("query", payload))
    trace_id = gen_trace_id()
    with trace("Manager run", trace_id=trace_id):
        print(f"Manager: starting run for query: {query}")
        # 1) call planner
        planner_res = await Runner.run(planner_agent, query)
        plan = planner_res.output
        if isinstance(plan, WebSearchPlan):
            searches = plan.searches
        elif isinstance(plan, dict) and "searches" in plan:
            searches = plan["searches"]
        else:
            raise RuntimeError("Planner did not return a WebSearchPlan")

        print(f"Manager: planner returned {len(searches)} searches, calling SearchAgent...")
        # 2) call search agent
        search_res = await Runner.run(search_agent, searches)
        search_results = search_res.output

        print(f"Manager: search returned {len(search_results)} items, calling WriterAgent...")
        # 3) call writer
        writer_payload = {"search_results": search_results}
        writer_res = await Runner.run(writer_agent, writer_payload)
        report = writer_res.output
        if not isinstance(report, ReportData):
            # try to coerce if dict-like
            if isinstance(report, dict) and "markdown_report" in report:
                report = ReportData(**report)
            else:
                raise RuntimeError("Writer did not return a ReportData")

        print("Manager: writing finished. Optionally sending email...")
        # 4) optional: send email (we'll send always in this demo)
        await Runner.run(email_agent, report)
        print("Manager: email sent. Returning report.")
        return report

register_agent_handler(manager_agent, _manager_handler)
