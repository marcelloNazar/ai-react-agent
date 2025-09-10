from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.agents.state import AgentState
from app.agents.nodes import assistant_node, tool_node, should_continue

def create_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("assistant", assistant_node)
    workflow.add_node("tools", tool_node)
    
    workflow.set_entry_point("assistant")
    
    workflow.add_conditional_edges(
        "assistant",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    workflow.add_edge("tools", "assistant")
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

graph = create_graph()
async_graph = create_graph()