from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from langchain_core.messages import BaseMessage

class ChatResponse(BaseModel):
    """Schema for chat responses."""
    
    thread_id: str = Field(..., description="Thread ID for the conversation")
    response: str = Field(..., description="The agent's response message")
    message_history: List[Dict[str, Any]] = Field(default_factory=list, description="History of messages in the conversation")
    tool_results: Dict[str, Any] = Field(default_factory=dict, description="Results from any tool calls")
    status: str = Field(default="completed", description="Status of the request")

class HealthResponse(BaseModel):
    """Schema for health check responses."""
    
    status: str = Field(default="healthy", description="Health status")

class StreamEvent(BaseModel):
    """Schema for streaming events."""
    
    event: str = Field(..., description="Type of event (message, tool_call, etc.)")
    data: Dict[str, Any] = Field(default_factory=dict, description="Event data")
    thread_id: Optional[str] = Field(None, description="Thread ID for the conversation")