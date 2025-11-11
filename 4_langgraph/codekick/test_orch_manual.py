import orchestrator
import graph

def set_agent_return(value):    
    class StubAgent:
        def invoke(self, prompt, **k):
            return value
    graph.agent = StubAgent()
    orchestrator.agent = graph.agent

print("\n----------------------------------------\n")

# 1) No tool
set_agent_return("I don't need tools")
print("no tool:", orchestrator.run_with_orchestration("hi"))

print("\n----------------------------------------\n")

# 2) Valid json string
set_agent_return('{"tool":"list_files","args":{}}')
orchestrator.TOOL_REGISTRY["list_files"] = {"fn": lambda: ["a","b"], "arg_model": None, "available": True}
print("list_files:", orchestrator.run_with_orchestration("list"))

print("\n----------------------------------------\n")

# 3) Unknown tool
set_agent_return('{"tool":"foo","args":{}}')
print("\n")
print("unknown:", orchestrator.run_with_orchestration("foo"))

print("\n----------------------------------------\n")

# 4) malformed
set_agent_return('{"tool":')
print("malformed:", orchestrator.run_with_orchestration("bad"))

print("\n----------------------------------------\n")


# 5) tool exception
def boom(**kwargs):
    raise RuntimeError("boom")
orchestrator.TOOL_REGISTRY["boom"] = {"fn": boom, "arg_model": None, "available": True}
set_agent_return('{"tool":"boom","args":{}}')
print("boom:", orchestrator.run_with_orchestration("boom"))

print("\n----------------------------------------\n")
