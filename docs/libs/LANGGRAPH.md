# LangGraph

## Visão Geral

LangGraph é um framework de orquestração de baixo nível para construir, gerenciar e implantar agentes de longa duração e com estado (stateful). Fornece execução durável, memória abrangente, capacidades human-in-the-loop e implantação pronta para produção.

**Principais características:**
- **Stateful**: Mantém estado entre execuções
- **Multi-Agent**: Suporte a sistemas multi-agente
- **Human-in-the-Loop**: Integração com aprovação humana
- **Persistência**: Checkpoints automáticos
- **Streaming**: Execução em tempo real
- **Condicional**: Fluxo baseado em condições

## Instalação

```bash
# Instalação básica
pip install langgraph

# Com LangChain
pip install langgraph "langchain[anthropic]"

# Para JavaScript/TypeScript
npm install @langchain/langgraph @langchain/core @langchain/anthropic
```

## Conceitos Fundamentais

### 1. StateGraph
```python
from langgraph.graph import StateGraph, MessagesState, START

# Definição do estado
class State(MessagesState):
    user_info: str = ""

# Criação do grafo
builder = StateGraph(State)
```

### 2. Nodes (Nós)
```python
def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}

# Adicionar nó ao grafo
builder.add_node("call_model", call_model)
```

### 3. Edges (Arestas)
```python
from langgraph.graph import START, END

# Aresta simples
builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)

# Aresta condicional
def should_continue(state):
    return "continue" if len(state["messages"]) < 10 else "end"

builder.add_conditional_edges(
    "call_model",
    should_continue,
    {
        "continue": "call_model",
        "end": END
    }
)
```

## Agentes React

### Criação Rápida
```python
from langgraph.prebuilt import create_react_agent

# Definir ferramenta
def get_weather(city: str) -> str:
    """Obtém informações do clima."""
    return f"Clima em {city}: Ensolarado, 25°C"

# Criar agente
agent = create_react_agent(
    model="anthropic:claude-3-5-sonnet-latest",
    tools=[get_weather],
    prompt="Você é um assistente útil"
)

# Executar
response = agent.invoke({
    "messages": [{"role": "user", "content": "Como está o clima em São Paulo?"}]
})
```

### Agente Personalizado
```python
from langgraph.prebuilt import tools_condition
from langgraph.graph import MessagesState

def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def tool_calling_agent():
    builder = StateGraph(MessagesState)
    
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tools))
    
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", "__end__": "__end__"}
    )
    builder.add_edge("tools", "agent")
    
    return builder.compile()
```

## Persistência e Checkpoints

### MemorySaver
```python
from langgraph.checkpoint.memory import InMemorySaver

# Checkpoint em memória
checkpointer = InMemorySaver()

graph = builder.compile(checkpointer=checkpointer)

# Configuração da thread
config = {"configurable": {"thread_id": "1"}}

# Execução com estado persistente
for chunk in graph.stream(
    {"messages": [{"role": "user", "content": "Olá!"}]},
    config,
    stream_mode="values"
):
    print(chunk["messages"][-1])
```

### PostgreSQL Checkpointer
```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

DB_URI = "postgresql://user:pass@localhost:5432/db"

async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # Configuração inicial (uma vez)
    # await checkpointer.setup()
    
    graph = builder.compile(checkpointer=checkpointer)
    
    async for chunk in graph.astream(
        {"messages": [{"role": "user", "content": "Olá!"}]},
        {"configurable": {"thread_id": "1"}},
        stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()
```

## Sistemas Multi-Agent

```python
from langgraph.prebuilt import create_react_agent

# Agente de voos
flight_assistant = create_react_agent(
    model="anthropic:claude-3-5-sonnet-latest",
    tools=[book_flight, transfer_to_hotel_assistant],
    prompt="Você é um assistente de reservas de voo",
    name="flight_assistant"
)

# Agente de hotéis
hotel_assistant = create_react_agent(
    model="anthropic:claude-3-5-sonnet-latest",
    tools=[book_hotel, transfer_to_flight_assistant],
    prompt="Você é um assistente de reservas de hotel",
    name="hotel_assistant"
)

# Grafo multi-agente
multi_agent_graph = (
    StateGraph(MessagesState)
    .add_node(flight_assistant)
    .add_node(hotel_assistant)
    .add_edge(START, "flight_assistant")
    .compile()
)
```

## Human-in-the-Loop

```python
from langgraph.graph import interrupt

def human_approval(state):
    # Interrompe a execução para aprovação humana
    interrupt("Aguardando aprovação para continuar...")
    return {"approved": True}

builder.add_node("approval", human_approval)
```

## Streaming

```python
# Stream de valores
for chunk in graph.stream(input_data, config, stream_mode="values"):
    print(chunk)

# Stream de updates
for chunk in graph.stream(input_data, config, stream_mode="updates"):
    print(chunk)

# Stream debug
for chunk in graph.stream(input_data, config, stream_mode="debug"):
    print(chunk)
```

## Configuração e Deploy

### Estrutura de Projeto
```
my-app/
├── my_agent/
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── tools.py
│   │   ├── nodes.py
│   │   └── state.py
│   ├── __init__.py
│   └── agent.py
├── .env
├── langgraph.json
└── pyproject.toml
```

### langgraph.json
```json
{
    "name": "meu-agente",
    "agent_type": "assistant",
    "configuration_schema": {
        "model": {
            "type": "string",
            "default": "anthropic:claude-3-5-sonnet-latest"
        }
    }
}
```

### Servidor Local
```bash
# Instalar em modo desenvolvimento
pip install -e .

# Iniciar servidor
langgraph dev

# Servidor estará disponível em http://localhost:8123
```

## LangGraph Studio

LangGraph Studio é uma interface visual para:
- Desenvolvimento visual de grafos
- Debug interativo
- Teste de fluxos
- Monitoramento em tempo real

```bash
# Iniciar servidor para o Studio
langgraph dev

# Acessar Studio em http://localhost:8123
```

## Casos de Uso

1. **Customer Support**: Sistemas de suporte automatizado
2. **Research Agents**: Agentes de pesquisa e análise
3. **Workflow Automation**: Automação de processos complexos
4. **Code Assistants**: Assistentes de programação
5. **Multi-step Planning**: Planejamento e execução de tarefas complexas

## Integração com o Projeto

No projeto atual, LangGraph pode ser usado para:
- Orquestração de agentes complexos
- Fluxos de trabalho multi-step
- Persistência de estado entre conversas
- Integração human-in-the-loop
- Sistemas de aprovação automática

## Exemplo Prático: Self-RAG

```python
from langgraph.graph import StateGraph, START, END

class RAGState(MessagesState):
    documents: list[str] = []
    relevance_score: float = 0.0

def retrieve_documents(state: RAGState):
    # Buscar documentos relevantes
    docs = retriever.get_relevant_documents(state["messages"][-1].content)
    return {"documents": docs}

def check_relevance(state: RAGState):
    # Verificar relevância dos documentos
    score = relevance_checker.score(state["documents"], state["messages"][-1])
    return {"relevance_score": score}

def generate_response(state: RAGState):
    # Gerar resposta baseada nos documentos
    response = model.invoke(state["messages"] + state["documents"])
    return {"messages": [response]}

# Construir grafo
builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve_documents)
builder.add_node("check", check_relevance)  
builder.add_node("generate", generate_response)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "check")
builder.add_conditional_edges(
    "check",
    lambda x: "generate" if x["relevance_score"] > 0.5 else "retrieve",
    {"generate": "generate", "retrieve": "retrieve"}
)
builder.add_edge("generate", END)

graph = builder.compile()
```

## Monitoramento e Observabilidade

```python
# Configurar LangSmith para observabilidade
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "sua_api_key"
os.environ["LANGCHAIN_PROJECT"] = "meu_projeto_langgraph"
```

## Links Úteis

- **Documentação Oficial**: https://langchain-ai.github.io/langgraph/
- **GitHub**: https://github.com/langchain-ai/langgraph
- **Studio**: https://studio.langchain.com/
- **PyPI**: https://pypi.org/project/langgraph/