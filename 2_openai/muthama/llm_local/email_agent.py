# email_agent.py
from agents import Agent, register_agent_handler, llm_call, _extract_json_from_text
from writer_agent import ReportData
import asyncio

email_agent = Agent(
    name="EmailAgent",
    instructions="Compose an email body from a markdown report and return a status string or JSON.",
    model="gpt-4o-mini",
    output_type=str,
)

def _simulated_email(payload):
    body = payload.markdown_report if isinstance(payload, ReportData) else (payload.get("markdown_report") if isinstance(payload, dict) else str(payload))
    print("=== Simulated Email ===")
    print("To: research@example.com")
    print("Subject: Your research report")
    print(body[:1000])
    print("=== End email ===")
    return "email_sent"

async def _email_handler(agent: Agent, payload):
    # payload can be ReportData or markdown string
    md = payload.markdown_report if isinstance(payload, ReportData) else (payload.get("markdown_report") if isinstance(payload, dict) else str(payload))
    system = "You are an Email agent. Given a markdown report, produce JSON {subject, body, status} or return 'email_sent'."
    user = f"Report markdown (truncated): {md[:2000]}\nReturn JSON with subject and body and status."

    try:
        text = await llm_call(system, user, model=agent.model)
        parsed = _extract_json_from_text(text)
        if isinstance(parsed, dict) and parsed.get("status"):
            # simulate sending by printing
            print("=== Email Send (LLM content) ===")
            print("Subject:", parsed.get("subject"))
            print(parsed.get("body")[:1000])
            return parsed.get("status")
        else:
            return _simulated_email(payload)
    except Exception:
        return _simulated_email(payload)

register_agent_handler(email_agent, _email_handler)
