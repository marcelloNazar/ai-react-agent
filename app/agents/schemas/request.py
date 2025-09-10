from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ChatRequest(BaseModel):
    """Schema for chat requests."""
    
    message: str = Field(..., description="The message to send to the agent")
    thread_id: Optional[str] = Field(None, description="Thread ID for conversation continuation")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context for the agent")

class StreamRequest(BaseModel):
    """Schema for streaming chat requests."""
    
    message: str = Field(..., description="The message to send to the agent")  
    thread_id: Optional[str] = Field(None, description="Thread ID for conversation continuation")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context for the agent")