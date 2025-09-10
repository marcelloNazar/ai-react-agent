from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.agents import route
from app.agents.schemas.response import HealthResponse

app = FastAPI(
    title="Python AI Agent API",
    description="AI Agent API with FastAPI, LangChain and LangGraph",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(route.router)

@app.get("/")
async def root():
    return {
        "message": "Python AI Agent API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)