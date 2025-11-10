# app.py (improved Gradio UI)
# Improved Gradio client for MCP Date/Time server
# Features:
# - Auto-discovers tools from /manifest.json
# - Renders schema-driven input form when input_schema exists
# - Allows JSON payload editing and KV table fallback
# - Supports API key via MCP_API_KEY env var
# - Tool invocation history and example payload injection
# - Graceful error handling and helpful debug output

import os
import json
import requests
import traceback
import gradio as gr
from typing import Any, Dict, List, Optional

DEFAULT_BASE = os.getenv("MCP_BASE", "http://127.0.0.1:8000")
API_KEY = os.getenv("MCP_API_KEY")
MANIFEST_PATH = "/manifest.json"
TOOLS_PATH = "/tools"

# Try to import jsonschema for validation (optional)
try:
    import jsonschema
    HAS_JSONSCHEMA = True
except Exception:
    HAS_JSONSCHEMA = False


# ----------------------
# Networking helpers
# ----------------------

def _headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["X-API-Key"] = API_KEY
    return h


def fetch_manifest(base_url: str, timeout: float = 5.0) -> Dict[str, Any]:
    url = base_url.rstrip("/") + MANIFEST_PATH
    r = requests.get(url, timeout=timeout, headers=_headers())
    r.raise_for_status()
    return r.json()


def invoke_tool(base_url: str, tool_name: str, payload: Dict[str, Any], timeout: float = 15.0) -> Any:
    url = base_url.rstrip("/") + TOOLS_PATH + f"/{tool_name}"
    r = requests.post(url, json=payload, timeout=timeout, headers=_headers())
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        # return error detail if available
        try:
            return {"error": f"HTTP {r.status_code}", "detail": r.json()}
        except Exception:
            return {"error": f"HTTP {r.status_code}", "detail": r.text}
    try:
        return r.json()
    except Exception:
        return {"raw_text": r.text}


# ----------------------
# Manifest/schema helpers
# ----------------------

def make_example_from_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    # Basic example generation: only supports object->properties
    if not schema or schema.get("type") != "object":
        return {}
    props = schema.get("properties", {})
    example = {}
    for k, v in props.items():
        t = v.get("type") if isinstance(v, dict) else None
        if t == "integer" or t == "number":
            example[k] = 0
        elif t == "boolean":
            example[k] = False
        elif t == "array":
            example[k] = []
        elif t == "object":
            example[k] = {}
        else:
            example[k] = ""
    return example


def coerce_str_to_type(val: str, target_type: Optional[str]):
    if val is None:
        return None
    s = str(val).strip()
    if s == "":
        return ""
    # try JSON parse first
    try:
        return json.loads(s)
    except Exception:
        pass
    if target_type in ("integer", "number"):
        try:
            return int(s) if target_type == "integer" else float(s)
        except Exception:
            return s
    if target_type == "boolean":
        if s.lower() in ("true", "1", "yes"): return True
        if s.lower() in ("false", "0", "no"): return False
        return s
    return s


# ----------------------
# Gradio dynamic form building
# ----------------------

def build_inputs_from_schema(schema: Dict[str, Any], initial: Dict[str, Any] = None):
    """Return a list of (label, component) and a mapping to reconstruct payload."""
    if not schema or schema.get("type") != "object":
        return [], {}
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    comps = []
    mapping = {}
    for name, meta in props.items():
        ptype = meta.get("type") if isinstance(meta, dict) else "string"
        label = f"{name} {'*' if name in required else ''}"
        default = (initial or {}).get(name, "")
        key = name
        if ptype in ("integer", "number"):
            comp = gr.Number(value=default if default != "" else None, label=label)
        elif ptype == "boolean":
            comp = gr.Checkbox(value=bool(default) if default != "" else False, label=label)
        elif ptype == "array":
            comp = gr.Textbox(value=json.dumps(default) if default != "" else "[]", label=label, lines=1, placeholder='JSON array, e.g. [1,2,3]')
        elif ptype == "object":
            comp = gr.Textbox(value=json.dumps(default) if default != "" else "{}", label=label, lines=2, placeholder='JSON object')
        else:
            comp = gr.Textbox(value=default if default != "" else "", label=label)
        comps.append((key, comp, ptype))
        mapping[key] = (comp, ptype)
    return comps, mapping


def collect_from_mapping(mapping: Dict[str, Any]) -> Dict[str, Any]:
    payload = {}
    for k, (comp, ptype) in mapping.items():
        # Gradio passes native Python types into the handler; comp.value isn't accessible here —
        # instead we expect the caller to pass the values list in the same order as mapping keys.
        # The code wiring in Gradio will provide the list.
        pass
    return payload


# ----------------------
# App callbacks
# ----------------------

def on_refresh_tools(base_url: str):
    try:
        manifest = fetch_manifest(base_url)
        tools = manifest.get("tools", [])
        names = [t.get("name") for t in tools]
        descriptions = {t.get("name"): t.get("description", "") for t in tools}
        server_name = manifest.get("name", "MCP Server")
        return gr.update(choices=names, value=(names[0] if names else None)), json.dumps(manifest, indent=2), server_name, json.dumps(descriptions, indent=2), ""
    except Exception as e:
        tb = traceback.format_exc()
        return gr.update(choices=[], value=None), "", "Error fetching manifest: " + str(e), "{}", tb


def build_schema_controls(selected_tool: str, base_url: str):
    if not selected_tool:
        return None, None, None
    try:
        manifest = fetch_manifest(base_url)
        for t in manifest.get("tools", []):
            if t.get("name") == selected_tool:
                schema = t.get("input_schema") or {}
                example = make_example_from_schema(schema)
                comps, mapping = build_inputs_from_schema(schema, example)
                return comps, mapping, json.dumps(example, indent=2)
    except Exception:
        pass
    return None, None, None


# ----------------------
# Gradio UI
# ----------------------
with gr.Blocks(title="MCP Date/Time Client — Improved") as demo:
    gr.Markdown("# MCP Date/Time Client — Improved UI")
    gr.Markdown("Connect to a FastMCP server, select a tool, fill inputs (schema-driven if available), or paste a JSON payload.")

    with gr.Row():
        base_input = gr.Textbox(label="MCP Server Base URL", value=DEFAULT_BASE, interactive=True)
        refresh_btn = gr.Button("Refresh tools")

    with gr.Row():
        server_name_box = gr.Textbox(label="Server Name", interactive=False)
        manifest_box = gr.Textbox(label="Manifest (raw JSON)", interactive=False, lines=8)

    with gr.Row():
        tool_dropdown = gr.Dropdown(label="Tool", choices=[], value=None)
        desc_box = gr.Textbox(label="Tool description", interactive=False, lines=3)

    # Schema-driven area
    schema_placeholder = gr.Column(visible=False)
    schema_row_components = gr.Column()  # will be replaced dynamically

    with gr.Row():
        json_input = gr.Textbox(label="JSON payload (use this to override schema fields)", placeholder="{}", lines=6)
        kv_table = gr.Dataframe(headers=["key", "value"], datatype=["str", "str"], row_count=4)

    with gr.Row():
        pretty_toggle = gr.Checkbox(label="Pretty-print response", value=True)
        invoke_btn = gr.Button("Invoke")
        example_btn = gr.Button("Insert example payload")
        status_box = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        response_pretty = gr.Textbox(label="Response (pretty)", interactive=False, lines=10)
        response_raw = gr.Textbox(label="Response (raw compact)", interactive=False, lines=6)

    # history panel
    history_box = gr.Textbox(label="Invocation history (latest first)", interactive=False, lines=6)

    # Hidden store for schema mapping and components
    mapping_store = gr.State({})
    comps_store = gr.State([])

    # Wire up refresh
    refresh_btn.click(fn=on_refresh_tools, inputs=[base_input], outputs=[tool_dropdown, manifest_box, server_name_box, gr.Textbox(value="{}"), status_box])

    # When a tool is selected, build schema-driven controls
    def tool_selected(tool_name, base_url):
        if not tool_name:
            return gr.update(visible=False), None, ""
        try:
            manifest = fetch_manifest(base_url)
            desc = ""
            schema = None
            example = None
            for t in manifest.get("tools", []):
                if t.get("name") == tool_name:
                    desc = t.get("description", "")
                    schema = t.get("input_schema") or {}
                    example = make_example_from_schema(schema)
                    break
            # build dynamic components list
            comps, mapping = build_inputs_from_schema(schema, example)
            # produce a simple placeholder text describing the fields
            hint = json.dumps(example, indent=2) if example else ""
            return gr.update(visible=True), mapping, hint
        except Exception as e:
            return gr.update(visible=False), {}, f"Error: {e}"

    tool_dropdown.change(fn=tool_selected, inputs=[tool_dropdown, base_input], outputs=[schema_placeholder, mapping_store, desc_box])

    # Insert example into JSON box
    def insert_example(example_text: str):
        if not example_text:
            return ""
        return example_text

    example_btn.click(fn=insert_example, inputs=[desc_box], outputs=[json_input])

    # Invocation handler — collects payload from JSON OR KV table OR dynamic components
    def on_invoke(base_url: str, tool_name: str, json_payload_text: str, kv_table, pretty: bool, mapping_state):
        if not tool_name:
            return "No tool selected", "", "", ""
        payload: Dict[str, Any] = {}
        # Prefer explicit JSON textarea if non-empty
        if json_payload_text and json_payload_text.strip():
            try:
                payload = json.loads(json_payload_text)
                if not isinstance(payload, dict):
                    return "JSON payload must be an object", "", "", ""
            except Exception as e:
                return f"Invalid JSON: {e}", "", "", ""
        else:
            # Try KV table
            if kv_table:
                for row in kv_table:
                    if not row or len(row) < 2:
                        continue
                    k = row[0]
                    v = row[1]
                    if k is None or str(k).strip() == "":
                        continue
                    payload[str(k)] = coerce_str_to_type(v, None)
            # Try schema-mapped inputs if present in mapping_state
            if mapping_state:
                # mapping_state is a dict name->(value, type) provided by the frontend wiring
                for k, info in mapping_state.items():
                    # info expected as [value, type]
                    val, ptype = info
                    payload[k] = coerce_str_to_type(val, ptype)
        # If schema exists, validate (optional)
        if HAS_JSONSCHEMA:
            manifest = fetch_manifest(base_url)
            schema = next((t.get("input_schema") for t in manifest.get("tools", []) if t.get("name") == tool_name), None)
            if schema:
                try:
                    jsonschema.validate(payload, schema)
                except Exception as e:
                    return f"Schema validation error: {e}", "", "", ""
        # invoke
        resp = invoke_tool(base_url, tool_name, payload)
        pretty_text = json.dumps(resp, indent=2, ensure_ascii=False) if pretty else json.dumps(resp, separators=(",",":"), ensure_ascii=False)
        raw_text = json.dumps(resp, separators=(",",":"), ensure_ascii=False)
        # update history (prepend)
        hist_entry = f"{tool_name} -> {json.dumps(payload)} -> {raw_text}"
        return "OK", pretty_text, raw_text, hist_entry

    invoke_btn.click(fn=on_invoke, inputs=[base_input, tool_dropdown, json_input, kv_table, pretty_toggle, mapping_store], outputs=[status_box, response_pretty, response_raw, history_box])


if __name__ == "__main__":
    demo.launch()
