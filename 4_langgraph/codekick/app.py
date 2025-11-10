import traceback
import gradio as gr
from graph import agent

def run_agent(user_prompt: str, recursion_limit: int):
    """
    Runs the existing agent.invoke call and returns a string suitable for display.
    """
    if not user_prompt:
        return "No prompt provided."

    try:
        result = agent.invoke(
            {"user_prompt": user_prompt},
            {"recursion_limit": recursion_limit}
        )
        return f"Final State:\n{result}"
    except Exception:
        return "Error:\n" + traceback.format_exc()

def build_ui():
    with gr.Blocks(title="Engineering Project Planner") as ui:
        gr.Markdown("## Engineering Project Planner")
        with gr.Row():
            prompt = gr.Textbox(
                lines=2,
                placeholder="Enter your project prompt...",
                label="Project prompt",
                elem_id="query_textbox"
            )
        with gr.Row():
            run_btn = gr.Button("Generate Project", elem_id="run_button")

        recursion = gr.Number(value=100, label="Recursion limit", precision=0)
        output = gr.Textbox(label="Output", interactive=False, lines=22)

        # Hook up button (also run when user presses enter in the textbox)
        run_btn.click(fn=run_agent, inputs=[prompt, recursion], outputs=output)
        prompt.submit(fn=run_agent, inputs=[prompt, recursion], outputs=output)

        return ui

if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=7860)
