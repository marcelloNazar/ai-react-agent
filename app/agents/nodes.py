from typing import Dict, Any
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode
from app.agents.state import AgentState
from app.config.llm import llm
from app.agents.tools.tools import get_tools

def assistant_node(state: AgentState) -> Dict[str, Any]:
    try:
        messages = state.get("messages", [])
        tools = get_tools()
        llm_with_tools = llm.bind_tools(tools)
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    except Exception as e:
        error_message = AIMessage(content=f"I encountered an error: {str(e)}")
        return {"messages": [error_message]}

tools = get_tools()
tool_node = ToolNode(tools)

def should_continue(state: AgentState) -> str:
    try:
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        
        if last_message and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        else:
            return "end"
    except Exception:
        return "end"