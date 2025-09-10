# HTTPX

## Visão Geral

HTTPX é um cliente HTTP de próxima geração para Python. É uma biblioteca que oferece suporte a requisições síncronas e assíncronas, HTTP/1.1 e HTTP/2, e é construída sobre conceitos familiares da biblioteca `requests`.

**Principais características:**
- **Requisições Síncronas e Assíncronas**: Suporte completo para ambos os paradigmas
- **HTTP/2**: Suporte nativo ao protocolo HTTP/2
- **Type Hints**: Totalmente tipado para melhor experiência de desenvolvimento
- **Streaming**: Suporte a streaming de responses
- **Timeouts**: Configuração flexível de timeouts
- **Autenticação**: Múltiplos métodos de autenticação
- **Compatibilidade**: API similar ao `requests`

## Instalação

```bash
# Instalação básica
pip install httpx

# Com suporte HTTP/2
pip install httpx[http2]

# Com suporte completo (HTTP/2, Brotli, SOCKS)
pip install httpx[all]
```

## Uso Básico

### Requisições Simples

```python
import httpx

# GET request
r = httpx.get('https://httpbin.org/get')
print(r.status_code)  # 200
print(r.text)

# POST request
r = httpx.post('https://httpbin.org/post', data={'key': 'value'})

# Outros métodos HTTP
r = httpx.put('https://httpbin.org/put', data={'key': 'value'})
r = httpx.delete('https://httpbin.org/delete')
r = httpx.head('https://httpbin.org/get')
r = httpx.options('https://httpbin.org/get')
```

### Parâmetros de URL

```python
# Parâmetros simples
params = {'key1': 'value1', 'key2': 'value2'}
r = httpx.get('https://httpbin.org/get', params=params)
print(r.url)  # https://httpbin.org/get?key1=value1&key2=value2

# Múltiplos valores para a mesma chave
params = {'key1': 'value1', 'key2': ['value2', 'value3']}
r = httpx.get('https://httpbin.org/get', params=params)
```

## Response Object

### Acessando Conteúdo

```python
r = httpx.get('https://www.example.org/')

# Conteúdo como texto
print(r.text)

# Conteúdo como bytes
print(r.content)

# JSON
r = httpx.get('https://api.github.com/events')
data = r.json()

# Status code
print(r.status_code)
print(r.status_code == httpx.codes.OK)  # True
```

### Headers e Cookies

```python
# Acessar headers
print(r.headers)
print(r.headers['Content-Type'])
print(r.headers.get('content-type'))

# Cookies da resposta
print(r.cookies)
```

## Enviando Dados

### Form Data

```python
# Dados como form-encoded
data = {'key1': 'value1', 'key2': 'value2'}
r = httpx.post("https://httpbin.org/post", data=data)

# Múltiplos valores
data = {'key1': ['value1', 'value2']}
r = httpx.post("https://httpbin.org/post", data=data)
```

### JSON

```python
# Enviar JSON
json_data = {'key': 'value', 'number': 42}
r = httpx.post("https://httpbin.org/post", json=json_data)
```

### Upload de Arquivos

```python
# Upload simples
with open('report.pdf', 'rb') as f:
    files = {'upload-file': f}
    r = httpx.post("https://httpbin.org/post", files=files)

# Com filename e content type
with open('report.pdf', 'rb') as f:
    files = {'upload-file': ('report.pdf', f, 'application/pdf')}
    r = httpx.post("https://httpbin.org/post", files=files)

# Com dados adicionais
data = {'message': 'Hello, world!'}
with open('report.pdf', 'rb') as f:
    files = {'file': f}
    r = httpx.post("https://httpbin.org/post", data=data, files=files)
```

## Headers Personalizados

```python
headers = {'user-agent': 'my-app/0.0.1'}
r = httpx.get('https://httpbin.org/headers', headers=headers)
```

## Cookies

```python
# Enviar cookies
cookies = {"peanut": "butter"}
r = httpx.get('https://httpbin.org/cookies', cookies=cookies)

# Gerenciamento avançado de cookies
cookies = httpx.Cookies()
cookies.set('cookie_on_domain', 'hello, there!', domain='httpbin.org')
r = httpx.get('http://httpbin.org/cookies', cookies=cookies)
```

## Redirecionamentos

```python
# Por padrão, redirecionamentos não são seguidos automaticamente
r = httpx.get('http://github.com/')
print(r.status_code)  # 301
print(r.history)  # []
print(r.next_request)  # <Request('GET', 'https://github.com/')>

# Seguir redirecionamentos
r = httpx.get('http://github.com/', follow_redirects=True)
print(r.url)  # https://github.com/
print(r.status_code)  # 200
print(r.history)  # [<Response [301 Moved Permanently]>]
```

## Streaming

### Streaming de Bytes

```python
with httpx.stream("GET", "https://www.example.com") as r:
    for data in r.iter_bytes():
        print(data)
```

### Streaming de Linhas

```python
with httpx.stream("GET", "https://www.example.com") as r:
    for line in r.iter_lines():
        print(line)
```

### Streaming Condicional

```python
with httpx.stream("GET", "https://www.example.com") as r:
    if int(r.headers['Content-Length']) < 1000000:  # 1MB
        r.read()
        print(r.text)
```

## Cliente HTTP

### Cliente Síncrono

```python
with httpx.Client() as client:
    r = client.get('https://example.com')
    print(r.status_code)

# Configuração do cliente
client = httpx.Client(
    base_url="https://api.example.com",
    headers={'Authorization': 'Bearer token'},
    timeout=10.0
)

with client:
    r = client.get('/users')
```

### Cliente Assíncrono

```python
import asyncio

async def fetch_data():
    async with httpx.AsyncClient() as client:
        r = await client.get('https://example.com')
        return r.json()

# Executar
data = asyncio.run(fetch_data())

# Cliente assíncrono configurado
async with httpx.AsyncClient(
    base_url="https://api.example.com",
    headers={'Authorization': 'Bearer token'},
    timeout=30.0
) as client:
    r = await client.get('/users')
```

## Autenticação

### Basic Auth

```python
# Método 1: tupla
r = httpx.get("https://example.com", auth=("username", "password"))

# Método 2: objeto Auth
auth = httpx.BasicAuth("username", "password")
r = httpx.get("https://example.com", auth=auth)
```

### Digest Auth

```python
auth = httpx.DigestAuth("username", "password")
r = httpx.get("https://example.com", auth=auth)
```

### Bearer Token

```python
headers = {'Authorization': 'Bearer your_token_here'}
r = httpx.get("https://api.example.com", headers=headers)
```

### Custom Auth

```python
class CustomAuth(httpx.Auth):
    def __init__(self, api_key):
        self.api_key = api_key
    
    def auth_flow(self, request):
        request.headers['X-API-Key'] = self.api_key
        yield request

auth = CustomAuth("your_api_key")
r = httpx.get("https://api.example.com", auth=auth)
```

## Timeouts

```python
# Timeout global
r = httpx.get('https://example.com', timeout=5.0)

# Timeouts específicos
timeout = httpx.Timeout(5.0, connect=10.0, read=5.0, write=5.0)
r = httpx.get('https://example.com', timeout=timeout)

# Cliente com timeout padrão
with httpx.Client(timeout=10.0) as client:
    r = client.get('https://example.com')
```

## Tratamento de Erros

```python
try:
    response = httpx.get("https://www.example.com/")
    response.raise_for_status()
except httpx.RequestError as exc:
    print(f"Erro na requisição {exc.request.url!r}.")
except httpx.HTTPStatusError as exc:
    print(f"Erro HTTP {exc.response.status_code} ao acessar {exc.request.url!r}.")
```

## Configuração Avançada

### Proxies

```python
# Proxy HTTP
proxies = {
    "http://": "http://proxy.example.com:8080",
    "https://": "https://proxy.example.com:8080",
}

with httpx.Client(proxies=proxies) as client:
    r = client.get("https://example.com")

# Proxy SOCKS
proxies = {
    "http://": "socks5://proxy.example.com:1080",
    "https://": "socks5://proxy.example.com:1080",
}
```

### SSL/TLS

```python
# Verificação personalizada de certificados
import ssl

context = ssl.create_default_context()
context.check_hostname = False
context.verify_mode = ssl.CERT_NONE

with httpx.Client(verify=context) as client:
    r = client.get("https://example.com")

# Cliente com certificado cliente
with httpx.Client(cert=("client.crt", "client.key")) as client:
    r = client.get("https://example.com")
```

### Limites de Conexão

```python
limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
client = httpx.Client(limits=limits)
```

## HTTP/2

```python
# Cliente com HTTP/2
with httpx.Client(http2=True) as client:
    r = client.get('https://example.com')
    print(r.http_version)  # 'HTTP/2'
```

## Casos de Uso

1. **APIs REST**: Cliente para consumir APIs RESTful
2. **Web Scraping**: Scraping de sites com suporte a sessões
3. **Microserviços**: Comunicação entre serviços
4. **Download de Arquivos**: Streaming de arquivos grandes
5. **Testes**: Mock de requisições HTTP em testes

## Integração com o Projeto

No projeto atual, HTTPX é usado para:
- Comunicação com APIs externas
- Requisições assíncronas em handlers FastAPI
- Testes de endpoints da aplicação (TestClient internamente usa HTTPX)
- Download/upload de arquivos
- Integração com serviços de terceiros

## Comparação com Requests

| Recurso | HTTPX | Requests |
|---------|--------|----------|
| Async/Await | ✅ | ❌ |
| HTTP/2 | ✅ | ❌ |
| Type Hints | ✅ | ❌ |
| Streaming | ✅ | ✅ |
| Sessões | ✅ (Client) | ✅ (Session) |
| API | Similar | Referência |

## Performance

```python
import asyncio
import httpx

# Requisições paralelas assíncronas
async def fetch_urls(urls):
    async with httpx.AsyncClient() as client:
        tasks = [client.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)
        return responses

urls = ['https://example.com'] * 10
responses = asyncio.run(fetch_urls(urls))
```

## Links Úteis

- **Documentação Oficial**: https://www.python-httpx.org/
- **GitHub**: https://github.com/encode/httpx
- **PyPI**: https://pypi.org/project/httpx/
- **Changelog**: https://github.com/encode/httpx/blob/master/CHANGELOG.md