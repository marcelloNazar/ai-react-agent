# 🎯 PLANO PARA DEMONSTRAÇÃO - AGENTE IA REACT

**INSTRUÇÃO INICIAL OBRIGATÓRIA:**

```
ultra-think
```

## 📋 CONTEXTO DA DEMONSTRAÇÃO

**Objetivo:** Implementar apenas a pasta `app/agents/` com um agente React usando LangChain e LangGraph, todo o resto (FastAPI, configs, testes) já estão prontos.

**Cenário:** Demonstrar como um agente de IA pode ser construído rapidamente do zero, focando apenas no core da lógica de agente.

---

## ⚡ PRÉ-REQUISITOS JÁ CONFIGURADOS

✅ **Estes arquivos/pastas JÁ EXISTEM e NÃO DEVEM SER ALTERADOS:**

- `app/main.py` - FastAPI configurado
- `app/config/llm.py` - LLM OpenAI configurado
- `tests/test_e2e.py` - testes E2E prontos
- `tests/conftest.py` - Fixtures de teste
- `requirements.txt` - Dependências
- `.env` - Variáveis de ambiente

---

## 🚀 IMPLEMENTAÇÃO AO VIVO - APENAS 4 ARQUIVOS

### **PASSO 1: Criar estrutura da pasta agents**

```bash
mkdir app/agents
mkdir app/agents/tools
```

---

### **PASSO 2: Criar `app/agents/__init__.py`**

```python
# Arquivo vazio - apenas para tornar agents um módulo Python
```

---

### **PASSO 3: Criar `app/agents/state.py`**

**CÓDIGO EXATO:**

```python
from typing import Dict, List, Any, TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    tool_results: Dict[str, Any]
    context: Dict[str, Any]
```

**Explicação para a audiência:**

- `AgentState`: Define o estado que o agente mantém durante a conversa
- `messages`: Histórico de mensagens com auto-merge
- `tool_results`: Resultados das ferramentas chamadas
- `context`: Contexto adicional da conversa

---

### **PASSO 4: Criar `app/agents/tools/__init__.py`**

```python
# Arquivo vazio - apenas para tornar tools um módulo Python
```

---

### **PASSO 5: Criar `app/agents/tools/tools.py`**

**CÓDIGO EXATO:**

```python
from langchain_core.tools import tool
import os
from dotenv import load_dotenv

load_dotenv()

@tool
def add(a: int, b: int) -> int:
    """Add two integers together."""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract two integers."""
    return a - b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

@tool
def divide(a: int, b: int) -> float:
    """Divide two integers."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

@tool
def web_search(query: str) -> str:
    """Search the web using Tavily API."""
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        return "Web search is not available - TAVILY_API_KEY not configured"

    try:
        from langchain_tavily import TavilySearch

        os.environ["TAVILY_API_KEY"] = tavily_api_key

        tavily_tool = TavilySearch(max_results=3)

        results = tavily_tool.invoke({"query": query})

        if isinstance(results, dict) and 'results' in results:
            search_results = results['results']
            return "\n\n".join([
                f"{result.get('title', 'No title')}\n{result.get('content', 'No content')}"
                for result in search_results[:3]
            ])
        return str(results)

    except Exception as e:
        return f"Web search failed: {str(e)}"

def get_tools():
    tools = [add, subtract, multiply, divide]
    if os.getenv("TAVILY_API_KEY"):
        tools.append(web_search)
    return tools
```

**Explicação para a audiência:**

- 4 ferramentas matemáticas básicas
- 1 ferramenta de busca web (opcional)
- Tratamento de erro para divisão por zero
- Carregamento condicional de ferramentas

---

### **PASSO 6: Criar `app/agents/nodes.py`**

**CÓDIGO EXATO:**

```python
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
```

**Explicação para a audiência:**

- `assistant_node`: Nó que processa mensagens com LLM
- `tool_node`: Nó que executa ferramentas chamadas
- `should_continue`: Decide se precisa chamar ferramentas ou terminar

---

### **PASSO 7: Criar `app/agents/graph.py`**

**CÓDIGO EXATO:**

```python
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
```

**Explicação para a audiência:**

- `StateGraph`: Grafo de estados do LangGraph
- Fluxo: assistant → (se precisa tools) → tools → assistant → end
- `MemorySaver`: Mantém contexto da conversa
- Versões síncrona e assíncrona

---

### **PASSO 8: Criar `app/agents/schemas/__init__.py`**

```python
# Arquivo vazio - apenas para tornar schemas um módulo Python
```

---

### **PASSO 9: Criar `app/agents/schemas/request.py`**

**CÓDIGO EXATO:**

```python
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
```

---

### **PASSO 10: Criar `app/agents/schemas/response.py`**

**CÓDIGO EXATO:**

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

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
```

---

### **PASSO 11: Criar `app/agents/route.py`**

**CÓDIGO EXATO:**

```python
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
```

---

### **PASSO 12: INTEGRAÇÃO COM MAIN.PY**

**Adicionar esta linha em `app/main.py`:**

```python
from app.agents import route

# E depois adicionar o router:
app.include_router(route.router)
```

**CÓDIGO COMPLETO DO `app/main.py`:**

```python
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
```

---

## 🧪 GERAÇÃO DOS TESTES E2E - AO VIVO

### **PASSO 13: CRIAR ARQUIVO DE TESTES DO AGENTE**

**Agora vamos gerar testes específicos para validar nosso agente!**

**Explicar para a audiência:**
"Agora que implementamos o agente, vamos criar um arquivo de testes separado e específico para validar toda a funcionalidade do agente. Isso mostra uma boa prática de organização modular dos testes."

**CRIAR o novo arquivo `tests/test_agent_e2e.py` com o seguinte conteúdo completo:**

```python
import pytest
import json
import asyncio
from fastapi.testclient import TestClient
from app.main import app
import httpx

class TestAgentE2E:
    """End-to-end tests specifically for AI Agent functionality."""

    def test_agent_chat_basic(self, client):
        """Test basic chat functionality."""
        payload = {
            "message": "Hello, how are you?",
            "context": {}
        }

        response = client.post("/agent/chat", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "thread_id" in data
        assert "response" in data
        assert "message_history" in data
        assert "status" in data
        assert data["status"] == "completed"

    def test_agent_chat_with_calculation(self, client):
        """Test chat with mathematical calculation."""
        payload = {
            "message": "What is 2 + 2?",
            "context": {}
        }

        response = client.post("/agent/chat", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "4" in data["response"] or "four" in data["response"].lower()
        assert data["status"] == "completed"

    def test_agent_chat_with_thread_continuation(self, client):
        """Test conversation continuation with thread_id."""
        # First message
        payload1 = {
            "message": "My name is Alice",
            "context": {}
        }

        response1 = client.post("/agent/chat", json=payload1)
        assert response1.status_code == 200

        data1 = response1.json()
        thread_id = data1["thread_id"]

        # Second message with same thread_id
        payload2 = {
            "message": "What is my name?",
            "thread_id": thread_id,
            "context": {}
        }

        response2 = client.post("/agent/chat", json=payload2)
        assert response2.status_code == 200

        data2 = response2.json()
        assert data2["thread_id"] == thread_id
        # The agent should remember the name from previous conversation
        assert "alice" in data2["response"].lower() or "Alice" in data2["response"]

    def test_agent_stream_endpoint(self, client):
        """Test streaming endpoint."""
        payload = {
            "message": "Tell me a short joke",
            "context": {}
        }

        response = client.post("/agent/stream", json=payload)
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Check that we get SSE formatted data
        content = response.text
        assert "data:" in content
        assert "event" in content

    def test_math_tools(self, client):
        """Test all mathematical operations."""
        test_cases = [
            ("What is 10 + 5?", "15"),
            ("Calculate 20 - 8", "12"),
            ("Multiply 6 by 7", "42"),
            ("Divide 100 by 4", "25")
        ]

        for question, expected_answer in test_cases:
            payload = {
                "message": question,
                "context": {}
            }

            response = client.post("/agent/chat", json=payload)
            assert response.status_code == 200

            data = response.json()
            assert expected_answer in data["response"]
            assert data["status"] == "completed"

    def test_division_by_zero_handling(self, client):
        """Test division by zero error handling."""
        payload = {
            "message": "What is 10 divided by 0?",
            "context": {}
        }

        response = client.post("/agent/chat", json=payload)
        assert response.status_code == 200

        data = response.json()
        # Should handle the error gracefully
        response_lower = data["response"].lower()
        assert any(keyword in response_lower for keyword in ["cannot", "error", "undefined", "not possible", "zero"])

    def test_invalid_request_format(self, client):
        """Test handling of invalid request formats."""
        # Missing required message field
        payload = {
            "context": {}
        }

        response = client.post("/agent/chat", json=payload)
        assert response.status_code == 422  # Validation error

    def test_empty_message(self, client):
        """Test handling of empty message."""
        payload = {
            "message": "",
            "context": {}
        }

        response = client.post("/agent/chat", json=payload)
        # Should still work, just with empty input
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_async_client_functionality(self, async_client):
        """Test async client functionality."""
        payload = {
            "message": "Hello from async client",
            "context": {}
        }

        response = await async_client.post("/agent/chat", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "response" in data
        assert data["status"] == "completed"

    def test_context_passing(self, client):
        """Test that context is properly passed to the agent."""
        context = {
            "user_preference": "concise answers",
            "session_info": "test_session"
        }

        payload = {
            "message": "Give me a brief explanation of AI",
            "context": context
        }

        response = client.post("/agent/chat", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "completed"

    def test_multiple_tool_calls(self, client):
        """Test requests that might require multiple tool calls."""
        payload = {
            "message": "Calculate 5 + 3 and then multiply the result by 2",
            "context": {}
        }

        response = client.post("/agent/chat", json=payload)
        assert response.status_code == 200

        data = response.json()
        # Should show the final result: (5+3)*2 = 16
        assert "16" in data["response"]
        assert data["status"] == "completed"

    def test_web_search_bh_weather(self, client):
        """Test web search for weather in Belo Horizonte."""
        payload = {
            "message": "How is the weather in Belo Horizonte, Brazil?",
            "context": {}
        }

        response = client.post("/agent/chat", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "completed"
        # Should contain weather information or indicate search availability
        response_lower = data["response"].lower()
        assert any(keyword in response_lower for keyword in ["weather", "temperature", "belo horizonte", "not available"])
```

**Explicar para a audiência após implementar:**
"Agora temos uma estrutura modular de testes:

- **`test_e2e.py`**: 2 testes básicos (health + root endpoints)
- **`test_agent_e2e.py`**: 12 testes específicos do agente
- **Total: 14 testes E2E** cobrindo:
  - ✅ Funcionalidade básica de chat
  - ✅ Cálculos matemáticos (4 operações)
  - ✅ Continuidade de conversa (memória)
  - ✅ Streaming Server-Sent Events
  - ✅ Tratamento de erros
  - ✅ Busca web (se configurada)
  - ✅ Validações de entrada
  - ✅ Clientes síncronos e assíncronos

**Vantagem da estrutura modular:** Separação clara de responsabilidades e facilita manutenção."

---

## 🧪 TESTANDO

### **PASSO 14: Validar antes de rodar - OBRIGATÓRIO**

**Verificar se todas as importações estão corretas:**

```bash
python -c "from app.agents.graph import graph; print('✅ Graph importado com sucesso')"
python -c "from app.agents.tools.tools import get_tools; print('✅ Tools importadas:', len(get_tools()), 'tools')"
python -c "from app.agents.schemas.request import ChatRequest; print('✅ Schemas importados')"
```

### **PASSO 15: RODAR TESTES E2E - CRÍTICO**

**OBRIGATÓRIO: Executar AMBOS os arquivos de teste com verbose e parar se falhar:**

```bash
# 1. Primeiro, verificar se pytest encontra TODOS os testes
pytest tests/ --collect-only

# 2. Rodar TODOS os testes com máximo de detalhes
pytest tests/ -v -s --tb=short

# 3. Alternativo: rodar arquivos separadamente
pytest tests/test_e2e.py -v -s        # 2 testes básicos
pytest tests/test_agent_e2e.py -v -s  # 12 testes do agente

# 4. Se algum teste falhar, rodar individual para debug
pytest tests/test_e2e.py::TestE2E::test_health_endpoint -v -s
pytest tests/test_agent_e2e.py::TestAgentE2E::test_agent_chat_basic -v -s
```

**RESULTADO ESPERADO (TODOS OS 14 TESTES - 2 ARQUIVOS):**

```
tests/test_e2e.py::TestE2E::test_health_endpoint PASSED                        [  7%]
tests/test_e2e.py::TestE2E::test_root_endpoint PASSED                          [ 14%]
tests/test_agent_e2e.py::TestAgentE2E::test_agent_chat_basic PASSED            [ 21%]
tests/test_agent_e2e.py::TestAgentE2E::test_agent_chat_with_calculation PASSED [ 28%]
tests/test_agent_e2e.py::TestAgentE2E::test_agent_chat_with_thread_continuation PASSED [ 35%]
tests/test_agent_e2e.py::TestAgentE2E::test_agent_stream_endpoint PASSED       [ 42%]
tests/test_agent_e2e.py::TestAgentE2E::test_math_tools PASSED                  [ 50%]
tests/test_agent_e2e.py::TestAgentE2E::test_division_by_zero_handling PASSED   [ 57%]
tests/test_agent_e2e.py::TestAgentE2E::test_invalid_request_format PASSED      [ 64%]
tests/test_agent_e2e.py::TestAgentE2E::test_empty_message PASSED               [ 71%]
tests/test_agent_e2e.py::TestAgentE2E::test_async_client_functionality PASSED  [ 78%]
tests/test_agent_e2e.py::TestAgentE2E::test_context_passing PASSED             [ 85%]
tests/test_agent_e2e.py::TestAgentE2E::test_multiple_tool_calls PASSED         [ 92%]
tests/test_agent_e2e.py::TestAgentE2E::test_web_search_bh_weather PASSED       [100%]

========================== 14 passed in XXs ===========================
```

**🎯 VALIDAÇÃO CRÍTICA:**

- ✅ **14 testes** (não 13, não 15 - exatamente 14)
- ✅ **TODOS PASSED** (nenhum FAILED/ERROR/SKIPPED)
- ✅ **Tempo razoável** (< 30s normalmente)
- ✅ **Sem warnings críticos** do pytest

**❌ SE NÃO DER ESSE RESULTADO EXATO:**

- PARE imediatamente a apresentação
- Use debug commands fornecidos
- Corrija o problema antes de continuar

### **PASSO 16: TROUBLESHOOTING - SE TESTES FALHAREM**

**❌ Erro comum 1: Import Error**

```bash
# Erro: ModuleNotFoundError: No module named 'app.agents'
# Solução: Verificar se __init__.py existem em todas as pastas
find app/agents -name "__init__.py"
# Deve retornar 3 arquivos: agents, tools, schemas
```

**❌ Erro comum 2: OpenAI API Error**

```bash
# Erro: openai.AuthenticationError
# Solução: Verificar .env
cat .env | grep OPENAI_API_KEY
# Deve ter valor válido (não test-key-for-testing em produção)
```

**❌ Erro comum 3: Tool Call Error**

```bash
# Erro: 'ToolNode' object has no attribute 'invoke'
# Solução: Verificar se langgraph está na versão correta
pip show langgraph
# Deve ser >= 0.6.7
```

**❌ Erro comum 4: Teste de matemática falha**

```bash
# Se test_math_tools falhar, rodar individual:
pytest tests/test_e2e.py::TestE2E::test_math_tools -v -s
# Verificar se LLM está chamando as tools corretamente
```

**❌ Erro comum 5: Streaming não funciona**

```bash
# Se test_agent_stream_endpoint falhar
# Verificar se async_graph está sendo criado
python -c "from app.agents.graph import async_graph; print('✅ Async graph OK')"
```

### **PASSO 17: DEBUG AVANÇADO - SE AINDA HOUVER PROBLEMAS**

**🔧 Comandos de debug para execução ao vivo:**

```bash
# 1. Verificar estrutura completa criada
tree app/agents
# ou
find app/agents -type f -name "*.py" | sort

# 2. Testar cada componente isoladamente
python -c "
from app.agents.state import AgentState
from app.agents.tools.tools import get_tools
from app.agents.nodes import assistant_node, should_continue
from app.agents.graph import graph
print('✅ Todos os componentes importados com sucesso')
print(f'✅ {len(get_tools())} tools carregadas')
print(f'✅ Graph compilado: {type(graph)}')
"

# 3. Testar endpoint básico primeiro
curl -X POST http://localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"hello","context":{}}'

# 4. Se der erro 500, verificar logs do uvicorn
# Rodar uvicorn sem --reload para ver erros completos
uvicorn app.main:app --port 8000
```

**🚨 REGRA CRÍTICA PARA APRESENTAÇÃO:**

- **NÃO CONTINUE** se qualquer teste falhar
- **SEMPRE DEBUG** o erro antes de prosseguir
- **RODAR TESTES** após cada grande mudança
- **VALIDAR IMPORTAÇÕES** antes de rodar testes

### **PASSO 18: Testar no Postman APÓS TODOS OS TESTES PASSAREM**

**Endpoint 1 - Cálculo:**

```json
POST http://localhost:8000/agent/chat
{
  "message": "What is 2 + 2?",
  "context": {}
}
```

**Endpoint 2 - Busca Web (se configurado):**

```json
POST http://localhost:8000/agent/chat
{
  "message": "How is the weather in Belo Horizonte, Brazil?",
  "context": {}
}
```

**Endpoint 3 - Streaming:**

```json
POST http://localhost:8000/agent/stream
{
  "message": "Calculate 10 * 5 and then add 3",
  "context": {}
}
```

---

## ⚠️ VALIDAÇÕES OBRIGATÓRIAS

Após implementação, verificar que:

1. ✅ `/health` retorna `{"status": "healthy"}`
2. ✅ Cálculos matemáticos funcionam
3. ✅ Divisão por zero é tratada corretamente
4. ✅ Streaming SSE funciona
5. ✅ Thread_id mantém contexto da conversa
6. ✅ Todos os 14 testes E2E passam

---

## 🗑️ ARQUIVOS A SEREM DELETADOS PARA DEMONSTRAÇÃO

**Para simular criação do zero, delete estes arquivos antes da palestra:**

```bash
# Deletar toda a pasta agents
rm -rf app/agents/

# Deletar linha do router em main.py (comentar):
# app.include_router(route.router)

# Deletar import em main.py (comentar):
# from app.agents import route
# from app.agents.schemas.response import HealthResponse

# E usar:
# from pydantic import BaseModel
# class HealthResponse(BaseModel):
#     status: str = "healthy"
```

**Manter estes arquivos:**

- ✅ `app/main.py` (sem as linhas do agents)
- ✅ `app/config/llm.py`
- ✅ `tests/test_e2e.py`
- ✅ `tests/conftest.py`
- ✅ `requirements.txt`
- ✅ `.env`
- ✅ `.gitignore`
- ✅ `pyproject.toml`

---

## 🎯 ROTEIRO DA APRESENTAÇÃO

1. **Mostrar estrutura atual** (sem pasta agents, apenas test_e2e.py com 2 testes)
2. **Explicar o objetivo** (criar agente React completo + testes modulares)
3. **Implementar pasta agents/** (passos 1-11: state, tools, nodes, graph, schemas, route)
4. **Integrar com main.py** (passo 12: adicionar imports e router)
5. **Criar testes específicos do agente** (passo 12B: novo arquivo test_agent_e2e.py)
6. **Validar importações** (passo 13: teste imports agents/)
7. **Rodar aplicação** (passo 14: uvicorn)
8. **Executar todos os 14 testes E2E** (passo 15: 2 básicos + 12 agente)
9. **Debug se necessário** (passos 16-17: troubleshooting)
10. **Demonstrar no Postman** (passo 18: chat + streaming)

**🏆 RESULTADO:** Sistema completo funcional com arquitetura modular!

**Demonstração completa:** 11 arquivos de código + 1 arquivo de testes + validação ao vivo!

**Tempo estimado:** 20-25 minutos

---

## 🚨 INSTRUÇÕES CRÍTICAS PARA IA - APRESENTAÇÃO AO VIVO

### **ANTES DE COMEÇAR - OBRIGATÓRIO:**

```bash
ultra-think
```

### **REGRAS DE OURO - NUNCA QUEBRAR:**

1. **🔍 SEMPRE VALIDAR CÓDIGO ANTES DE IMPLEMENTAR**

   - Compare com código funcional existente
   - Verifique compatibilidade de importações
   - Teste importações antes de criar arquivos grandes

2. **🧪 TESTES SÃO OBRIGATÓRIOS**

   - NUNCA pule a validação de testes
   - Se 1 teste falhar = PARE e debugue
   - Todos os 14 testes DEVEM passar para considerar sucesso

3. **📝 IMPLEMENTAÇÃO PASSO A PASSO**

   - Implemente 1 arquivo por vez
   - Teste importações após cada arquivo
   - Valide estrutura antes de prosseguir

4. **🔧 DEBUG IMEDIATO**

   - Qualquer erro = para tudo e debuga
   - Use os comandos de debug fornecidos
   - Não continue com erros pendentes

5. **💬 COMUNICAÇÃO CLARA**
   - Explique cada conceito (AgentState, tools, LangGraph)
   - Mostre logs e resultados
   - Destaque arquitetura modular

### **CHECKLIST DE VALIDAÇÃO FINAL:**

- [ ] Pasta `app/agents/` criada com todos os arquivos (11 arquivos Python)
- [ ] Arquivo `tests/test_agent_e2e.py` criado (12 testes do agente)
- [ ] Estrutura modular: 2 arquivos de teste (test_e2e.py + test_agent_e2e.py)
- [ ] Todas as importações funcionando (sem erro)
- [ ] Aplicação roda sem erro (`uvicorn app.main:app`)
- [ ] Endpoint `/health` e `/` respondem corretamente
- [ ] **TODOS os 14 testes E2E passando** (2 + 12 = 14 total)
- [ ] Postman funciona com cálculos e streaming
- [ ] Demonstração completa: código + testes + validação ao vivo

### **SE ALGO DER ERRADO:**

- **PARE imediatamente**
- **Use comandos de debug do plano**
- **Corrija antes de continuar**
- **Re-teste tudo após correção**

**LEMBRE-SE:** Esta é uma demonstração AO VIVO. Erros são normais, mas devem ser debugados na hora para mostrar o processo real de desenvolvimento.

---

## 🗑️ PREPARAÇÃO FINAL - ARQUIVOS PARA DELETAR

### **COMANDOS PARA LIMPEZA PRÉ-APRESENTAÇÃO:**

```bash
# 1. DELETAR pasta agents (será recriada)
rm -rf app/agents/

# 2. DELETAR arquivos desnecessários
rm PLANO_SIMPLES_AGENTS.md
rm SIMPLIFICACAO_RESUMO.md
rm README.md
rm "=4.1.0"
rm -rf .pytest_cache/ .claude/

# 3. Verificar estrutura final
ls -la
```

### **MODIFICAR `app/main.py`:**

**REMOVER estas linhas:**

```python
from app.agents import route
from app.agents.schemas.response import HealthResponse
app.include_router(route.router)
```

**ADICIONAR no lugar:**

```python
from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str = "healthy"
```

### **ESTRUTURA FINAL PARA DEMONSTRAÇÃO:**

```bash
ai-react-agent/
├── app/
│   ├── __init__.py
│   ├── main.py                      # SEM agents (preparado)
│   └── config/
│       ├── __init__.py
│       └── llm.py                   # LLM configurado ✅
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # Setup completo ✅
│   └── test_e2e.py                  # APENAS 2 testes básicos ✅
├── requirements.txt                  # Dependências ✅
├── pyproject.toml                   # Config projeto ✅
├── .env                             # Vars ambiente ✅
├── .gitignore                       # Git ignore ✅
└── PLANO_DEMONSTRACAO_AO_VIVO.md    # ESTE PLANO ✅
```

### **🎯 VALIDAÇÃO PRÉ-APRESENTAÇÃO:**

```bash
# 1. Testar se aplicação roda (só endpoints básicos)
uvicorn app.main:app --port 8000

# 2. Testar endpoints básicos
curl http://localhost:8000/health  # {"status":"healthy"}
curl http://localhost:8000/        # {"status":"running"}

# 3. Rodar APENAS os 2 testes básicos
pytest tests/test_e2e.py -v       # 2 passed

# 4. Verificar estrutura de testes limpa
ls tests/                          # deve ter: __init__.py, conftest.py, test_e2e.py
ls tests/test_agent_e2e.py        # should error (não existe ainda)

# 5. Verificar que pasta agents NÃO existe
ls app/agents/                     # should error (será criada na demo)

# 6. Verificar que main.py NÃO tem imports de agents
grep "from app.agents" app/main.py # should error (já deve estar limpo)
```

### **📝 CHECKLIST FINAL PRÉ-APRESENTAÇÃO:**

- [ ] Pasta `app/agents/` deletada ✅
- [ ] `main.py` sem imports de agents ✅
- [ ] `test_e2e.py` com apenas 2 testes básicos limpos ✅
- [ ] `test_agent_e2e.py` NÃO existe (será criado na demo) ✅
- [ ] Aplicação roda e responde endpoints básicos ✅
- [ ] Apenas 2 testes básicos passam (pytest tests/test_e2e.py) ✅
- [ ] Estrutura modular limpa pronta para demonstração completa ✅

**AGORA ESTÁ PRONTO PARA DEMONSTRAÇÃO COMPLETA AO VIVO! 🚀**
