# FastAPI

## Visão Geral

FastAPI é um framework web moderno e rápido para construir APIs com Python 3.6+ baseado em type hints padrão do Python. É projetado para ser fácil de usar, rápido para codificar e pronto para produção.

**Principais características:**
- **Rápido**: Alta performance, equiparável ao NodeJS e Go
- **Rápido para codificar**: Aumenta a velocidade de desenvolvimento em 200% a 300%
- **Menos bugs**: Reduz cerca de 40% de erros humanos
- **Intuitivo**: Excelente suporte do editor com autocompletar
- **Fácil**: Projetado para ser fácil de usar e aprender
- **Curto**: Minimiza duplicação de código
- **Robusto**: Obtém código pronto para produção com documentação automática interativa
- **Baseado em padrões**: Baseado e totalmente compatível com os padrões abertos para APIs: OpenAPI e JSON Schema

## Instalação

```bash
# Instalação com dependências padrão (recomendado)
pip install "fastapi[standard]"

# Instalação mínima
pip install fastapi

# Com uv
uv add "fastapi[standard]"
```

## Uso Básico

### Aplicação Simples

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}
```

### Executando a Aplicação

```bash
# Modo desenvolvimento (com auto-reload)
fastapi dev main.py

# Ou com uvicorn
uvicorn main:app --reload
```

## Recursos Principais

### 1. Validação Automática
FastAPI usa Pydantic para validação automática de dados de entrada e saída.

### 2. Documentação Automática
- **Swagger UI**: Disponível em `/docs`
- **ReDoc**: Disponível em `/redoc`
- **OpenAPI Schema**: Disponível em `/openapi.json`

### 3. Type Hints
Usa type hints do Python para:
- Validação de dados
- Serialização
- Documentação automática
- Autocompletar no editor

### 4. Dependency Injection
Sistema poderoso de injeção de dependências para reutilização de código.

### 5. Segurança
- Integração com OAuth2
- Suporte a JWT tokens
- API keys
- Cookies seguros

## Testes

```python
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}
```

### Instalação das dependências de teste

```bash
pip install httpx pytest
```

## Deploy

### Uvicorn (Desenvolvimento)
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Gunicorn + Uvicorn (Produção)
```bash
pip install "uvicorn[standard]" gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
```

### Docker
```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Casos de Uso Comuns

1. **APIs REST**: Criação rápida de APIs RESTful
2. **Microserviços**: Desenvolvimento de microserviços
3. **APIs de Machine Learning**: Servir modelos ML
4. **APIs GraphQL**: Integração com GraphQL
5. **WebSockets**: Comunicação em tempo real

## Integração com o Projeto

No projeto atual, FastAPI é usado como framework principal para:
- Criação de endpoints REST
- Integração com agentes LangChain/LangGraph
- Validação de dados com Pydantic
- Documentação automática da API
- Estrutura assíncrona para performance

## Links Úteis

- **Documentação Oficial**: https://fastapi.tiangolo.com/
- **GitHub**: https://github.com/tiangolo/fastapi
- **PyPI**: https://pypi.org/project/fastapi/