# Python-dotenv

## Visão Geral

Python-dotenv é uma biblioteca que lê pares chave-valor de um arquivo `.env` e pode defini-los como variáveis de ambiente. Ajuda no desenvolvimento de aplicações seguindo os princípios dos 12 fatores, separando configuração do código.

**Principais características:**
- **Carregamento de .env**: Carrega variáveis de arquivos .env
- **Não-invasivo**: Não modifica o ambiente por padrão (com `dotenv_values()`)
- **Flexível**: Suporte a múltiplos arquivos .env
- **CLI**: Interface de linha de comando para gerenciar arquivos .env
- **IPython Support**: Integração com Jupyter notebooks

## Instalação

```bash
# Instalação básica
pip install python-dotenv

# Com suporte CLI
pip install "python-dotenv[cli]"
```

## Uso Básico

### Arquivo .env

Crie um arquivo `.env` na raiz do seu projeto:

```bash
# .env
DATABASE_URL=postgresql://localhost/mydatabase
SECRET_KEY=your_secret_key_here
DEBUG=True
API_KEY=abc123
EMAIL_USER=user@example.com
EMAIL_PASSWORD=password123
```

### Carregando Variáveis

```python
from dotenv import load_dotenv
import os

# Carrega as variáveis do arquivo .env
load_dotenv()

# Agora você pode acessar as variáveis
database_url = os.getenv('DATABASE_URL')
secret_key = os.getenv('SECRET_KEY')
debug = os.getenv('DEBUG') == 'True'
```

### Parsing para Dicionário

```python
from dotenv import dotenv_values

# Carrega em um dicionário sem modificar o ambiente
config = dotenv_values(".env")  
# config = {"DATABASE_URL": "postgresql://...", "SECRET_KEY": "..."}

# Usando o dicionário
database_url = config['DATABASE_URL']
```

## Múltiplos Arquivos .env

```python
import os
from dotenv import dotenv_values

# Combinação de múltiplos arquivos
config = {
    **dotenv_values(".env.shared"),    # variáveis compartilhadas
    **dotenv_values(".env.secret"),    # variáveis sensíveis
    **os.environ,                      # sobrescreve com variáveis do sistema
}
```

## Carregamento de Stream

```python
from io import StringIO
from dotenv import load_dotenv

# Carregamento de uma string
config = StringIO("DATABASE_URL=sqlite:///test.db\nDEBUG=True")
load_dotenv(stream=config)
```

## Interface CLI

```bash
# Instalar com CLI
pip install "python-dotenv[cli]"

# Definir variáveis
dotenv set DATABASE_URL postgresql://localhost/mydb
dotenv set SECRET_KEY my_secret_key
dotenv set DEBUG True

# Listar variáveis
dotenv list

# Listar em formato JSON
dotenv list --format=json

# Executar comando com variáveis carregadas
dotenv run -- python manage.py runserver
dotenv run -- pytest
```

## Integração com IPython/Jupyter

```python
# Em uma célula do Jupyter notebook
%load_ext dotenv
%dotenv

# Ou especificar um arquivo
%dotenv path/to/your/.env

# Opções disponíveis:
# -o: sobrescrever variáveis existentes
# -v: modo verboso
%dotenv -o -v
```

## Formatos Suportados

### Valores Simples

```bash
# .env
NAME=John Doe
AGE=30
ACTIVE=true
```

### Valores com Espaços

```bash
# Com aspas
MESSAGE="Hello World"
DESCRIPTION='A simple description'

# Sem aspas (não recomendado)
TITLE=My Application Title
```

### Valores Multilinhas

```bash
# Usando \n explícito
MULTILINE_TEXT="First line\nSecond line\nThird line"

# Multilinhas com aspas triplas
LONG_TEXT="""
This is a long text
that spans multiple lines
and preserves formatting
"""
```

### Variáveis sem Valor

```bash
# Variável vazia
EMPTY_VAR=

# Variável sem valor (será None em dotenv_values)
UNDEFINED_VAR
```

### Comentários

```bash
# Este é um comentário
DATABASE_URL=postgresql://localhost/mydb  # Comentário inline

# Configurações de desenvolvimento
DEBUG=True
LOG_LEVEL=DEBUG
```

## Configurações Avançadas

### Arquivo Específico

```python
from dotenv import load_dotenv

# Carregamento de arquivo específico
load_dotenv('.env.production')
load_dotenv('/path/to/custom/.env')
```

### Controle de Sobreposição

```python
# Não sobrescrever variáveis existentes
load_dotenv(override=False)

# Sobrescrever variáveis existentes (padrão)
load_dotenv(override=True)
```

### Busca de Arquivo

```python
# Busca arquivo .env subindo na hierarquia de diretórios
load_dotenv(dotenv_path=None, verbose=True)

# Especificar diretório de busca
from dotenv import find_dotenv
load_dotenv(find_dotenv())
```

## Integração com Frameworks

### FastAPI

```python
from fastapi import FastAPI
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

@app.get("/")
async def root():
    return {
        "database_url": os.getenv("DATABASE_URL"),
        "debug": os.getenv("DEBUG") == "True"
    }
```

### Django

```python
# settings.py
from dotenv import load_dotenv
import os

load_dotenv()

SECRET_KEY = os.getenv('SECRET_KEY')
DEBUG = os.getenv('DEBUG') == 'True'
DATABASE_URL = os.getenv('DATABASE_URL')

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.getenv('DB_NAME'),
        'USER': os.getenv('DB_USER'),
        'PASSWORD': os.getenv('DB_PASSWORD'),
        'HOST': os.getenv('DB_HOST', 'localhost'),
        'PORT': os.getenv('DB_PORT', '5432'),
    }
}
```

### Flask

```python
from flask import Flask
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['DATABASE_URL'] = os.getenv('DATABASE_URL')

@app.route('/')
def hello():
    return f"Debug mode: {os.getenv('DEBUG')}"
```

## Boas Práticas

### Estrutura de Arquivos

```
projeto/
├── .env.example          # Template público
├── .env                  # Desenvolvimento (não committar)
├── .env.local           # Configurações locais
├── .env.production      # Produção (não committar)
└── .gitignore           # Incluir .env*
```

### .env.example

```bash
# .env.example - Template para outros desenvolvedores
DATABASE_URL=postgresql://localhost/mydb
SECRET_KEY=your_secret_key_here
DEBUG=True
API_KEY=your_api_key
EMAIL_USER=your_email@example.com
EMAIL_PASSWORD=your_email_password
```

### .gitignore

```gitignore
# Environment variables
.env
.env.local
.env.production
.env.staging

# Mas manter o exemplo
!.env.example
```

### Validação de Variáveis

```python
from dotenv import load_dotenv
import os

load_dotenv()

def get_env_var(name, default=None, required=True):
    value = os.getenv(name, default)
    if required and value is None:
        raise ValueError(f"Environment variable {name} is required")
    return value

# Uso
DATABASE_URL = get_env_var('DATABASE_URL')
SECRET_KEY = get_env_var('SECRET_KEY')
DEBUG = get_env_var('DEBUG', 'False').lower() == 'true'
PORT = int(get_env_var('PORT', '8000'))
```

### Configuração com Classes

```python
from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY')
    DATABASE_URL = os.getenv('DATABASE_URL')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    @staticmethod
    def validate():
        required_vars = ['SECRET_KEY', 'DATABASE_URL']
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")

class DevelopmentConfig(Config):
    DEBUG = True
    
class ProductionConfig(Config):
    DEBUG = False

# Usar configuração baseada no ambiente
config_class = DevelopmentConfig if os.getenv('FLASK_ENV') == 'development' else ProductionConfig
```

## Casos de Uso

1. **Configuração de Aplicações**: Separar configuração sensível do código
2. **Diferentes Ambientes**: Dev, staging, produção
3. **Chaves de API**: Armazenar chaves sensíveis fora do código
4. **Configuração de Banco**: URLs de conexão com banco de dados
5. **Feature Flags**: Controlar funcionalidades via environment

## Integração com o Projeto

No projeto atual, python-dotenv é usado para:
- Carregamento de variáveis de ambiente da aplicação
- Configuração de chaves de API (OpenAI, Anthropic)
- Settings de desenvolvimento vs produção
- Configuração de banco de dados
- Parâmetros de configuração do sistema

## Exemplo Prático Completo

```python
# config.py
from dotenv import load_dotenv
import os
from typing import Optional

load_dotenv()

class Settings:
    def __init__(self):
        self.database_url: str = self._get_required('DATABASE_URL')
        self.secret_key: str = self._get_required('SECRET_KEY')
        self.debug: bool = os.getenv('DEBUG', 'False').lower() == 'true'
        self.port: int = int(os.getenv('PORT', '8000'))
        self.openai_api_key: Optional[str] = os.getenv('OPENAI_API_KEY')
        self.log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    
    def _get_required(self, key: str) -> str:
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Environment variable {key} is required")
        return value

# Instância global
settings = Settings()

# main.py
from config import settings
from fastapi import FastAPI

app = FastAPI(debug=settings.debug)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.port)
```

## Links Úteis

- **Documentação Oficial**: https://python-dotenv.readthedocs.io/
- **GitHub**: https://github.com/theskumar/python-dotenv
- **PyPI**: https://pypi.org/project/python-dotenv/
- **12 Factor App**: https://12factor.net/config