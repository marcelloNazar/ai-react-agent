# Uvicorn

## Visão Geral

Uvicorn é um servidor ASGI (Asynchronous Server Gateway Interface) ultrarrápido para Python. É uma implementação baseada em uvloop e httptools, oferecendo excelente performance para aplicações assíncronas, especialmente FastAPI.

**Principais características:**
- **Ultra Performance**: Baseado em uvloop e httptools
- **ASGI Compliant**: Suporte completo ao padrão ASGI
- **WebSocket Support**: Suporte nativo a WebSockets
- **HTTP/1.1 e HTTP/2**: Protocolos modernos
- **Hot Reload**: Recarga automática durante desenvolvimento
- **SSL/TLS**: Suporte a HTTPS

## Instalação

```bash
# Instalação mínima
pip install uvicorn

# Instalação com dependências padrão (recomendado)
pip install "uvicorn[standard]"

# Com uv
uv add uvicorn
uv add "uvicorn[standard]"
```

### Dependências Extras

- **uvloop**: Loop de eventos de alta performance (Linux/macOS)
- **httptools**: Parser HTTP otimizado
- **websockets**: Suporte aprimorado a WebSockets
- **watchfiles**: Monitoramento de arquivos para reload
- **colorama**: Logs coloridos (Windows)
- **python-dotenv**: Carregamento de variáveis de ambiente

## Uso Básico

### Aplicação ASGI Simples

```python
# main.py
async def app(scope, receive, send):
    assert scope['type'] == 'http'
    
    body = f'Received {scope["method"]} request to {scope["path"]}'
    
    await send({
        'type': 'http.response.start',
        'status': 200,
        'headers': [
            [b'content-type', b'text/plain'],
        ]
    })
    
    await send({
        'type': 'http.response.body',
        'body': body.encode('utf-8'),
    })
```

### Executando o Servidor

```bash
# Comando básico
uvicorn main:app

# Com host e porta específicos
uvicorn main:app --host 0.0.0.0 --port 8000

# Modo desenvolvimento (com reload)
uvicorn main:app --reload

# Com múltiplos workers
uvicorn main:app --workers 4
```

## Integração com FastAPI

### Aplicação FastAPI

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```

### Executando

```bash
# Desenvolvimento
uvicorn main:app --reload

# Produção
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Execução Programática

### Método uvicorn.run()

```python
import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    uvicorn.run("main:app", port=5000, log_level="info")
```

### Método Config + Server

```python
import uvicorn
from fastapi import FastAPI

app = FastAPI()

if __name__ == "__main__":
    config = uvicorn.Config("main:app", port=5000, log_level="info")
    server = uvicorn.Server(config)
    server.run()
```

### Execução Assíncrona

```python
import asyncio
import uvicorn
from fastapi import FastAPI

app = FastAPI()

async def main():
    config = uvicorn.Config("main:app", port=5000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuração Avançada

### Application Factory

```python
def create_app():
    app = FastAPI()
    
    @app.get("/")
    async def root():
        return {"message": "Hello from factory"}
    
    return app
```

```bash
# Executar com factory
uvicorn --factory main:create_app
```

### Configuração SSL/TLS

```bash
# HTTPS
uvicorn main:app --ssl-keyfile=./key.pem --ssl-certfile=./cert.pem

# Com configuração customizada
uvicorn main:app --ssl-keyfile=./key.pem --ssl-certfile=./cert.pem --ssl-ca-certs=./ca.pem
```

### Variables de Ambiente

```bash
# .env file
UVICORN_HOST=0.0.0.0
UVICORN_PORT=8000
UVICORN_LOG_LEVEL=info
UVICORN_WORKERS=4
```

```python
import os
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("UVICORN_HOST", "127.0.0.1"),
        port=int(os.getenv("UVICORN_PORT", "8000")),
        log_level=os.getenv("UVICORN_LOG_LEVEL", "info"),
        workers=int(os.getenv("UVICORN_WORKERS", "1"))
    )
```

## Deploy em Produção

### Com Gunicorn

```bash
# Instalar Gunicorn
pip install "uvicorn[standard]" gunicorn

# Executar com worker Uvicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app

# Com configuração específica
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000 --timeout 120
```

### Worker Personalizado

```python
from uvicorn.workers import UvicornWorker

class MyUvicornWorker(UvicornWorker):
    CONFIG_KWARGS = {
        "loop": "asyncio",
        "http": "httptools",
        "lifespan": "on",
        "access_log": True,
    }
```

### Docker

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Instalar dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar aplicação
COPY . .

# Expor porta
EXPOSE 8000

# Comando de inicialização
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - UVICORN_HOST=0.0.0.0
      - UVICORN_PORT=8000
      - UVICORN_WORKERS=4
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Logging e Monitoramento

### Configuração de Log

```python
import logging
import uvicorn

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        log_level="info",
        access_log=True,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        }
    )
```

### Health Check

```python
from fastapi import FastAPI, status

app = FastAPI()

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }
```

## WebSocket Support

```python
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message text was: {data}")

@app.get("/")
async def get():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Chat</title>
        </head>
        <body>
            <h1>WebSocket Chat</h1>
            <form action="" onsubmit="sendMessage(event)">
                <input type="text" id="messageText" autocomplete="off"/>
                <button>Send</button>
            </form>
            <ul id='messages'>
            </ul>
            <script>
                var ws = new WebSocket("ws://localhost:8000/ws");
                ws.onmessage = function(event) {
                    var messages = document.getElementById('messages')
                    var message = document.createElement('li')
                    var content = document.createTextNode(event.data)
                    message.appendChild(content)
                    messages.appendChild(message)
                };
                function sendMessage(event) {
                    var input = document.getElementById("messageText")
                    ws.send(input.value)
                    input.value = ''
                    event.preventDefault()
                }
            </script>
        </body>
    </html>
    """)
```

## Performance Tuning

### Configurações Recomendadas

```bash
# Para aplicações CPU-intensive
uvicorn main:app --workers 4 --loop asyncio

# Para aplicações I/O-intensive  
uvicorn main:app --workers 8 --loop uvloop

# Com keep-alive personalizado
uvicorn main:app --timeout-keep-alive 5

# Com limite de requisições simultâneas
uvicorn main:app --limit-concurrency 1000
```

### Otimizações

```python
# Configuração otimizada
config = uvicorn.Config(
    app="main:app",
    host="0.0.0.0", 
    port=8000,
    workers=4,
    loop="uvloop",  # Linux/macOS
    http="httptools",
    timeout_keep_alive=5,
    limit_concurrency=1000,
    backlog=2048
)
```

## Casos de Uso

1. **APIs REST**: Servidor para APIs FastAPI/Starlette
2. **WebSocket Apps**: Aplicações real-time
3. **Microserviços**: Serviços de alta performance
4. **Proxy Reverso**: Behind nginx ou load balancers
5. **Development Server**: Servidor de desenvolvimento

## Integração com o Projeto

No projeto atual, Uvicorn é usado como:
- Servidor ASGI principal para FastAPI
- Desenvolvimento com hot-reload
- Produção com múltiplos workers
- Suporte a conexões assíncronas
- Base para deploy em containers

## Alternativas

- **Hypercorn**: Suporte HTTP/2 e HTTP/3
- **Daphne**: Servidor ASGI do Django Channels
- **Gunicorn**: Process manager (usado com Uvicorn workers)

## Comandos Úteis

```bash
# Verificar versão
uvicorn --version

# Ajuda completa
uvicorn --help

# Executar com configuração específica
uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-level debug

# Múltiplos workers com reload desabilitado  
uvicorn main:app --workers 4 --host 0.0.0.0 --port 8000
```

## Links Úteis

- **Documentação Oficial**: https://www.uvicorn.org/
- **GitHub**: https://github.com/encode/uvicorn
- **PyPI**: https://pypi.org/project/uvicorn/
- **ASGI Specification**: https://asgi.readthedocs.io/