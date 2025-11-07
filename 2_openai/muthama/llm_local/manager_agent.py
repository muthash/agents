# manager_agent.py
from typing import Any
from agents import Agent, Runner, trace, gen_trace_id, register_agent_handler
from planner_agent import planner_agent, WebSearchPlan, WebSearchItem
from search_agent import search_agent, SearchResultItem
from writer_agent import writer_agent, ReportData
from email_agent import email_agent

INSTRUCTIONS = (
    "You are the Research Manager and may call PlannerAgent, SearchAgent, WriterAgent, EmailAgent. "
    "Coordinate them to produce a ReportData for the input query. This runtime will call the tools "
    "using Runner.run; simply orchestrate the flow by letting the orchestrator call tools in order."
)

manager_agent = Agent(
    name="ManagerAgent",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    agents_as_tools=[planner_agent, search_agent, writer_agent, email_agent],
    handoffs={
        "to_planner": {"tool_name": "PlannerAgent", "expects": "str"},
        "to_search": {"tool_name": "SearchAgent", "expects": "list[WebSearchItem]"},
        "to_writer": {"tool_name": "WriterAgent", "expects": "list[SearchResultItem]"},
        "to_email": {"tool_name": "EmailAgent", "expects": "ReportData or str"},
        "done": {"tool_name": None, "expects": "ReportData"},
    },
    output_type=ReportData,
)

# Manager uses Runner.run to call tools (manager itself can be simple)
async def _manager_handler(agent: Agent, payload: Any):
    query = payload if isinstance(payload, str) else str(payload.get("query", payload))
    trace_id = gen_trace_id()
    with trace("Manager run", trace_id=trace_id):
        print(f"Manager: starting run for query: {query}")
        planner_res = await Runner.run(planner_agent, query)
        plan = planner_res.output
        if isinstance(plan, WebSearchPlan):
            searches = plan.searches
        elif isinstance(plan, dict) and "searches" in plan:
            searches = [WebSearchItem(**s) if isinstance(s, dict) else s for s in plan["searches"]]
        else:
            raise RuntimeError("Planner did not return a WebSearchPlan")

        search_res = await Runner.run(search_agent, searches)
        search_results = search_res.output

        writer_payload = {"search_results": search_results}
        writer_res = await Runner.run(writer_agent, writer_payload)
        report = writer_res.output
        if not isinstance(report, ReportData):
            if isinstance(report, dict) and "markdown_report" in report:
                report = ReportData(**report)
            else:
                raise RuntimeError("Writer did not return a ReportData")

        await Runner.run(email_agent, report)
        return report

register_agent_handler(manager_agent, _manager_handler)
