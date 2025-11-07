# email_agent.py
from agents import Agent, register_agent_handler
from writer_agent import ReportData

email_agent = Agent(
    name="EmailAgent",
    instructions="Send an email containing the given markdown report (simulated).",
    model="simulated-email-model",
    output_type=str,
)

def _email_handler(agent: Agent, payload):
    # payload may be a ReportData or markdown string
    body = None
    if isinstance(payload, ReportData):
        body = payload.markdown_report
    elif isinstance(payload, dict) and "markdown_report" in payload:
        body = payload["markdown_report"]
    else:
        body = str(payload)
    # simulate sending
    print("=== Simulated Email Sending ===")
    print("To: research@example.com")
    print("Subject: Your research report")
    print()
    print(body[:1000])  # print first 1000 chars
    print("=== Email sent (simulated) ===")
    return "email_sent"

register_agent_handler(email_agent, _email_handler)
