# deep_research.py
import asyncio
import sys
from manager_agent import manager_agent
from agents import Runner

async def main(argv):
    if len(argv) < 2:
        print("Usage: python deep_research.py \"your query here\"")
        return
    query = argv[1]
    result = await Runner.run(manager_agent, query)
    report = result.output
    print("\n--- Final Report ---\n")
    if hasattr(report, "markdown_report"):
        print(report.markdown_report)
    else:
        print(report)

if __name__ == "__main__":
    asyncio.run(main(sys.argv))