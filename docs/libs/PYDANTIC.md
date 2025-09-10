# Pydantic

## Visão Geral

Pydantic é uma biblioteca de validação de dados para Python que usa type hints para definir esquemas de dados, oferecendo validação rápida e extensível para aplicações. É amplamente utilizada em APIs, especialmente com FastAPI.

**Principais características:**
- **Validação Rápida**: Performance otimizada em Rust (Pydantic V2)
- **Type Safety**: Segurança de tipos baseada em type hints
- **Serialização**: Conversão automática entre tipos Python e JSON/dict
- **Extensível**: Validadores customizados e tipos personalizados
- **IDE Support**: Excelente suporte a autocompletar e type checking

## Instalação

```bash
# Instalação básica
pip install pydantic

# Com suporte a email
pip install 'pydantic[email]'

# Com suporte completo (email, timezone)
pip install 'pydantic[email,timezone]'

# Usando uv
uv add pydantic
uv add 'pydantic[email,timezone]'

# Via conda
conda install pydantic -c conda-forge
```

## Uso Básico

### BaseModel

```python
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, PositiveInt

class User(BaseModel):
    id: int
    name: str = 'John Doe'
    signup_ts: datetime | None = None
    tastes: dict[str, PositiveInt]

# Validação e conversão automática
external_data = {
    'id': 123,
    'signup_ts': '2019-06-01 12:22',
    'tastes': {
        'wine': 9,
        b'cheese': 7,  # bytes convertido para str
        'cabbage': '1',  # string convertida para int
    },
}

user = User(**external_data)
print(user.id)  # 123
print(user.model_dump())  # Serialização para dict
```

### Validação de Dados

```python
from pydantic import BaseModel, ValidationError

class Person(BaseModel):
    name: str
    age: int

# Validação bem-sucedida
person = Person(name="Alice", age=30)

# Validação com erro
try:
    person = Person(name="Bob", age="not_a_number")
except ValidationError as e:
    print(e)
    # age: Input should be a valid integer
```

## Field e Validadores

### Field Configuration

```python
from pydantic import BaseModel, Field
from typing import Optional

class Item(BaseModel):
    name: str = Field(
        min_length=1,
        max_length=50,
        description="Nome do item"
    )
    price: float = Field(
        gt=0,  # greater than
        le=1000,  # less than or equal
        description="Preço em reais"
    )
    description: Optional[str] = Field(
        None,
        max_length=200,
        description="Descrição opcional"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags do item"
    )
```

### Validadores Customizados

```python
from pydantic import BaseModel, field_validator, model_validator

class UserAccount(BaseModel):
    username: str
    password: str
    password_confirm: str

    @field_validator('username')
    @classmethod
    def username_must_be_alphanumeric(cls, v):
        if not v.isalnum():
            raise ValueError('Username deve ser alfanumérico')
        return v

    @model_validator(mode='after')
    def check_passwords_match(self):
        if self.password != self.password_confirm:
            raise ValueError('Senhas não coincidem')
        return self
```

## Tipos Especiais

### Built-in Types

```python
from pydantic import BaseModel, EmailStr, HttpUrl, UUID4
from datetime import datetime
from uuid import uuid4

class ComplexModel(BaseModel):
    # Tipos básicos
    email: EmailStr
    website: HttpUrl
    uuid: UUID4
    created_at: datetime
    
    # Tipos numéricos com validação
    positive_int: int = Field(gt=0)
    percentage: float = Field(ge=0, le=100)
    
    # Estruturas
    tags: set[str]
    metadata: dict[str, str]
```

### Enum Integration

```python
from enum import Enum
from pydantic import BaseModel

class Color(str, Enum):
    RED = "red"
    GREEN = "green"  
    BLUE = "blue"

class Car(BaseModel):
    brand: str
    color: Color
    year: int = Field(ge=1900, le=2024)

car = Car(brand="Toyota", color="red", year=2023)
```

## Serialização e Parsing

### Model Methods

```python
user = User(id=1, name="Alice", tastes={"coffee": 5})

# Serialização
user_dict = user.model_dump()
user_json = user.model_dump_json()

# Parsing
user_from_dict = User.model_validate(user_dict)
user_from_json = User.model_validate_json(user_json)
user_from_strings = User.model_validate_strings({
    'id': '1', 
    'name': 'Alice'
})
```

### Exclusão e Inclusão

```python
# Excluir campos
user.model_dump(exclude={'tastes'})

# Incluir apenas campos específicos
user.model_dump(include={'id', 'name'})

# Excluir campos não definidos
user.model_dump(exclude_unset=True)
```

## Configuração de Modelo

### Model Config

```python
from pydantic import BaseModel, ConfigDict

class StrictModel(BaseModel):
    model_config = ConfigDict(
        # Não permite campos extras
        extra='forbid',
        # Validação estrita (sem coerção de tipos)
        strict=True,
        # Permite mutação após criação
        frozen=False,
        # Validação de atribuição
        validate_assignment=True
    )
    
    name: str
    age: int
```

### Aliases

```python
class ApiResponse(BaseModel):
    user_name: str = Field(alias='userName')
    is_active: bool = Field(alias='isActive')
    
    model_config = ConfigDict(
        # Usar aliases na serialização
        populate_by_name=True
    )

# Uso com aliases
data = {'userName': 'Alice', 'isActive': True}
response = ApiResponse(**data)
```

## Settings Management

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "My App"
    debug: bool = False
    database_url: str
    secret_key: str
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )

# Carrega automaticamente do ambiente/.env
settings = Settings()
```

## JSON Schema

```python
from pydantic import BaseModel

class Address(BaseModel):
    street: str
    city: str
    zipcode: str

class Person(BaseModel):
    name: str
    age: int
    address: Address

# Gerar JSON Schema
schema = Person.model_json_schema()
print(schema)
# Retorna schema OpenAPI/JSON Schema completo
```

## Integração com FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class CreateUserRequest(BaseModel):
    name: str
    email: EmailStr
    age: int = Field(ge=18)

class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    created_at: datetime

@app.post("/users/", response_model=UserResponse)
async def create_user(user: CreateUserRequest):
    # Validação automática do request body
    # Serialização automática do response
    return UserResponse(
        id=1,
        name=user.name,
        email=user.email,
        created_at=datetime.now()
    )
```

## Error Handling

```python
from pydantic import ValidationError

try:
    user = User(id="invalid", name="")
except ValidationError as e:
    print(e.json())  # JSON formatado
    
    # Acesso a erros individuais
    for error in e.errors():
        print(f"Campo: {error['loc']}")
        print(f"Mensagem: {error['msg']}")
        print(f"Tipo: {error['type']}")
```

## Performance e Otimização

### Strict Mode
```python
# Desabilita coerção de tipos para melhor performance
class FastModel(BaseModel):
    model_config = ConfigDict(strict=True)
    
    id: int
    name: str
```

### Reused Validator
```python
from pydantic import BaseModel, field_validator
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_validation(value: str) -> str:
    # Validação custosa com cache
    return value.upper()

class CachedModel(BaseModel):
    name: str
    
    @field_validator('name')
    def validate_name(cls, v):
        return expensive_validation(v)
```

## Casos de Uso Comuns

1. **API Validation**: Validação de request/response em APIs
2. **Configuration**: Gerenciamento de configurações de aplicação
3. **Data Processing**: Validação e transformação de dados
4. **CLI Tools**: Validação de argumentos de linha de comando
5. **Database Models**: Validação antes de persistir dados

## Integração com o Projeto

No projeto atual, Pydantic é usado para:
- Validação de dados de entrada na API FastAPI
- Definição de esquemas de request/response
- Configuração de settings da aplicação
- Validação de dados dos agentes LangChain
- Serialização/deserialização JSON

## Migração V1 → V2

```bash
# Ferramenta de migração automática
pip install bump-pydantic
bump-pydantic meu_projeto/
```

## Ferramentas de Desenvolvimento

### MyPy Integration
```ini
# mypy.ini
[mypy]
plugins = pydantic.mypy

[pydantic-mypy]
init_forbid_extra = True
init_typed = True
warn_required_dynamic_aliases = True
```

### Code Generation
```bash
# Gerar modelos Pydantic de JSON Schema
pip install datamodel-code-generator
datamodel-codegen --input schema.json --output models.py
```

## Links Úteis

- **Documentação Oficial**: https://docs.pydantic.dev/
- **GitHub**: https://github.com/pydantic/pydantic
- **PyPI**: https://pypi.org/project/pydantic/
- **Pydantic Settings**: https://docs.pydantic.dev/latest/concepts/pydantic_settings/