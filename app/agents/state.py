from typing import Dict, List, Any, TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    tool_results: Dict[str, Any]
    context: Dict[str, Any]