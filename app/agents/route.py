from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from app.agents.schemas.request import ChatRequest, StreamRequest
from app.agents.schemas.response import ChatResponse, StreamEvent
from app.agents.graph import graph, async_graph
from app.agents.state import AgentState
import uuid

router = APIRouter(prefix="/agent", tags=["agent"])

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        thread_id = request.thread_id or str(uuid.uuid4())
        
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }
        
        initial_state: AgentState = {
            "messages": [HumanMessage(content=request.message)],
            "tool_results": {},
            "context": request.context or {}
        }
        
        result = graph.invoke(initial_state, config)
        
        messages = result.get("messages", [])
        last_message = messages[-1] if messages else None
        
        if not last_message:
            raise HTTPException(status_code=500, detail="No response generated")
        
        response_content = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        message_history = []
        for msg in messages:
            if hasattr(msg, 'content'):
                message_history.append({
                    "role": msg.__class__.__name__,
                    "content": msg.content
                })
        
        return ChatResponse(
            thread_id=thread_id,
            response=response_content,
            message_history=message_history,
            tool_results=result.get("tool_results", {}),
            status="completed"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/stream")
async def stream_chat(request: StreamRequest):
    try:
        thread_id = request.thread_id or str(uuid.uuid4())
        
        async def event_generator():
            try:
                config = {
                    "configurable": {
                        "thread_id": thread_id
                    }
                }
                
                initial_state: AgentState = {
                    "messages": [HumanMessage(content=request.message)],
                    "tool_results": {},
                    "context": request.context or {}
                }
                
                start_event = StreamEvent(
                    event="start",
                    data={"thread_id": thread_id},
                    thread_id=thread_id
                )
                yield f"data: {start_event.model_dump_json()}\n\n"
                
                async for chunk in async_graph.astream(initial_state, config):
                    for node, output in chunk.items():
                        if "messages" in output:
                            messages = output["messages"]
                            for message in messages:
                                if hasattr(message, 'content') and message.content:
                                    message_event = StreamEvent(
                                        event="message",
                                        data={
                                            "content": message.content,
                                            "type": message.__class__.__name__
                                        },
                                        thread_id=thread_id
                                    )
                                    yield f"data: {message_event.model_dump_json()}\n\n"
                                
                                if hasattr(message, 'tool_calls') and message.tool_calls:
                                    for tool_call in message.tool_calls:
                                        tool_event = StreamEvent(
                                            event="tool_call",
                                            data={
                                                "tool": tool_call.get("name", "unknown"),
                                                "args": tool_call.get("args", {})
                                            },
                                            thread_id=thread_id
                                        )
                                        yield f"data: {tool_event.model_dump_json()}\n\n"
                
                end_event = StreamEvent(
                    event="complete",
                    data={"status": "completed"},
                    thread_id=thread_id
                )
                yield f"data: {end_event.model_dump_json()}\n\n"
                
            except Exception as e:
                error_event = StreamEvent(
                    event="error",
                    data={"error": str(e)},
                    thread_id=thread_id
                )
                yield f"data: {error_event.model_dump_json()}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")