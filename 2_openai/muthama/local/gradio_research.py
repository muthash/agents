# deep_research.py
# Gradio UI for the research manager flow.
# This file replaces the CLI entrypoint and runs the ManagerAgent via Runner.run,
# displaying the resulting markdown report in a Gradio interface.
#
# Usage:
#   - Ensure your project files are in place (agents.py, planner_agent.py, search_agent.py,
#     writer_agent.py, email_agent.py, manager_agent.py).
#   - Optionally set OPENAI_API_KEY if you want LLM-backed agents.
#   - Run: python deep_research.py
#
# The UI exposes a query textbox and returns a markdown report and a status message.

import asyncio
import traceback
import gradio as gr

from agents import Runner, gen_trace_id, trace
from manager_agent import manager_agent

# Gradio-friendly async runner
async def _run_manager_async(query: str):
    trace_id = gen_trace_id()
    status_lines = []
    def log(s: str):
        status_lines.append(s)
        # return the combined status as plain text so the caller can update UI
        return "\n".join(status_lines)

    with trace("DeepResearch-run", trace_id=trace_id):
        log(f"[TRACE] id={trace_id}")
        try:
            log("Starting manager agent...")
            result = await Runner.run(manager_agent, query)
            out = result.output
            # If manager returned a ReportData-like object, extract markdown_report
            markdown = None
            try:
                # some implementations return a dataclass-like or dict
                if hasattr(out, "markdown_report"):
                    markdown = out.markdown_report
                elif isinstance(out, dict) and "markdown_report" in out:
                    markdown = out["markdown_report"]
                else:
                    # fallback: stringify output
                    markdown = str(out)
            except Exception:
                markdown = str(out)
            log("Manager finished successfully.")
            return markdown, log("Finished.")
        except Exception as e:
            tb = traceback.format_exc()
            log(f"Error: {e}")
            log(tb)
            # return error information in the status and a short message for markdown output
            return f"**Error running research manager:** {e}", log("Failed with exception. See status for traceback.")

# Gradio wrapper that calls the async function
def run_query(query: str):
    # Run the async manager in the current event loop if available, otherwise start a new one.
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an existing event loop (gradio runs one); schedule coroutine and wait.
        coro = _run_manager_async(query)
        task = asyncio.ensure_future(coro)
        # Return results by awaiting task synchronously via utils from gradio.
        # gradio supports async functions directly, but some environments call sync wrapper.
        # To keep compatibility, we wait for completion (this is safe inside gradio).
        # NOTE: If your gradio version supports async directly, you could pass _run_manager_async as the callback.
        while not task.done():
            # yield control briefly to the event loop
            import time
            time.sleep(0.05)
        return task.result()
    else:
        # No running loop, start one
        return asyncio.run(_run_manager_async(query))

# Build Gradio UI
with gr.Blocks(title="Deep Research Manager") as demo:
    gr.Markdown("# Deep Research Manager")
    with gr.Row():
        query_in = gr.Textbox(label="Research query", placeholder="e.g. Latest trends in performance testing for microservices", lines=2)
    with gr.Row():
        run_btn = gr.Button("Run Research")
        clear_btn = gr.Button("Clear")
    with gr.Row():
        output_md = gr.Markdown("", label="Report")
    with gr.Row():
        status_box = gr.Textbox(label="Status / Trace log", interactive=False, lines=8)

    # Wire up events
    def _on_run(q):
        # run_query returns (markdown, status)
        try:
            md, status = run_query(q)
            return md, status
        except Exception as e:
            tb = traceback.format_exc()
            return f"**Error:** {e}", tb

    run_btn.click(fn=_on_run, inputs=[query_in], outputs=[output_md, status_box])
    clear_btn.click(fn=lambda: ("", ""), inputs=None, outputs=[output_md, status_box])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
