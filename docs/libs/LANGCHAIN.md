# LangChain

## Visão Geral

LangChain é um framework para desenvolvimento de aplicações alimentadas por modelos de linguagem grandes (LLMs). Simplifica todas as etapas do ciclo de vida de aplicações LLM, oferecendo componentes de código aberto e integrações com terceiros.

**Principais características:**
- **Modular**: Componentes reutilizáveis e composáveis
- **Flexível**: Suporte a múltiplos provedores de LLM
- **Observável**: Ferramentas para monitoramento e debugging
- **Escalável**: De protótipos a aplicações de produção

## Instalação

```bash
# Instalação básica
pip install langchain

# Com integrações OpenAI
pip install langchain-openai

# Com comunidade (integrações extras)
pip install langchain-community

# Instalação completa para desenvolvimento
pip install langchain langchain-community langchain-openai
```

## Componentes Principais

### 1. Chat Models
```python
from langchain.chat_models import init_chat_model

# Configuração do modelo
model = init_chat_model(
    "anthropic:claude-3-5-sonnet-latest",
    temperature=0
)

# Uso direto
response = model.invoke("Explique o conceito de IA")
```

### 2. Prompts
```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Template de prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente útil especializado em {topic}."),
    MessagesPlaceholder("examples"),
    ("human", "{input}")
])
```

### 3. Output Parsers
```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="Nome da pessoa")
    age: int = Field(description="Idade da pessoa")

parser = PydanticOutputParser(pydantic_object=Person)
```

### 4. Chains
```python
from langchain_core.runnables import RunnablePassthrough

# Chain simples
chain = prompt | model | parser

# Execução
result = chain.invoke({
    "topic": "tecnologia", 
    "input": "Quem foi Alan Turing?",
    "examples": []
})
```

## Recursos Avançados

### Memory
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
```

### Vector Stores
```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Configuração do vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(
    texts=["texto1", "texto2"],
    embeddings=embeddings
)

# Busca por similaridade
docs = vectorstore.similarity_search("consulta")
```

### Tools e Agents
```python
from langchain.tools import tool
from langchain.agents import create_openai_functions_agent

@tool
def get_weather(city: str) -> str:
    """Obtém informações do clima para uma cidade."""
    return f"Clima em {city}: Ensolarado, 25°C"

# Criação do agent
agent = create_openai_functions_agent(
    llm=model,
    tools=[get_weather],
    prompt=prompt
)
```

## RAG (Retrieval-Augmented Generation)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# 1. Carregamento de documentos
loader = TextLoader("documento.txt")
documents = loader.load()

# 2. Split em chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
texts = text_splitter.split_documents(documents)

# 3. Criação do vector store
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings
)

# 4. Retriever
retriever = vectorstore.as_retriever()

# 5. Chain RAG
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=retriever
)
```

## Few-Shot Learning

```python
# Exemplos para few-shot
examples = [
    {"input": "2 + 2", "output": "4"},
    {"input": "3 * 3", "output": "9"}
]

# Seletor de exemplos
from langchain_core.example_selectors import SemanticSimilarityExampleSelector

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    embeddings,
    vectorstore,
    k=2
)
```

## Observabilidade

### LangSmith
```python
import os

# Configuração do LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "sua_api_key"
os.environ["LANGCHAIN_PROJECT"] = "meu_projeto"
```

## Casos de Uso

1. **Chatbots**: Assistentes conversacionais inteligentes
2. **Q&A Systems**: Sistemas de perguntas e respostas sobre documentos
3. **Text Summarization**: Resumo automático de textos
4. **Code Generation**: Geração e explicação de código
5. **Data Analysis**: Análise de dados usando linguagem natural

## Integração com o Projeto

No projeto atual, LangChain é usado para:
- Gerenciamento de modelos LLM (OpenAI)
- Criação de chains de processamento
- Integração com ferramentas externas
- Sistema de memória para conversas
- RAG para consulta de documentos

## Configuração de Ambiente

```python
import os
from getpass import getpass

# Configuração das chaves de API
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass("OpenAI API Key: ")

if "ANTHROPIC_API_KEY" not in os.environ:
    os.environ["ANTHROPIC_API_KEY"] = getpass("Anthropic API Key: ")
```

## Melhores Práticas

1. **Gestão de Tokens**: Monitore uso e custos dos LLMs
2. **Error Handling**: Implemente tratamento robusto de erros
3. **Caching**: Use cache para reduzir chamadas repetitivas
4. **Observabilidade**: Configure logging e tracing
5. **Testing**: Teste chains e components isoladamente

## Links Úteis

- **Documentação Oficial**: https://python.langchain.com/
- **GitHub**: https://github.com/langchain-ai/langchain
- **LangSmith**: https://smith.langchain.com/
- **PyPI**: https://pypi.org/project/langchain/