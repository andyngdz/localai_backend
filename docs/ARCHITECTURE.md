# Architecture

**Stack:** FastAPI + SQLAlchemy 2.0 + Socket.IO | SQLite database | Pytest + pytest-asyncio

**Structure:**

- `app/features/` - Feature modules (generators, downloads, models, histories, etc.)
- `app/cores/` - Core services (model_manager, model_loader, samplers, constants)
- `app/services/` - Utilities (device, image, storage, logger, memory)
- `app/database/` - SQLAlchemy models and CRUD operations

**Key conventions:**

- Tab indentation, single quotes (ruff.toml)
- Async/await throughout
- Pydantic schemas for all API responses (never raw dicts)
- Type hints required (pyright enforced)

**Pre-commit validation:** Husky runs `uv run ruff format`, `uv run ruff check`, and `uv run pyright` on staged files. Commits are blocked if any check fails.

## Avoiding Circular Imports

**Use shared schemas in `app/schemas/` to avoid circular dependencies.** When multiple features need to reference the same data structure, create a shared schema module instead of importing between features.

**Bad example (circular import):**

```python
# app/cores/model_manager/pipeline_manager.py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.features.generators.schemas import LoRAData

def load_loras(self, lora_configs: list['LoRAData']) -> None:
    # Using string literal forward reference as workaround
    pass
```

**Good example (shared schemas):**

```python
# app/schemas/lora.py (shared location)
from pydantic import BaseModel, Field

class LoRAData(BaseModel):
    id: int = Field(..., description='Database ID')
    name: str = Field(..., description='Display name')
    weight: float = Field(..., ge=0.0, le=2.0)

# app/cores/model_manager/pipeline_manager.py
from app.schemas.lora import LoRAData

def load_loras(self, lora_configs: list[LoRAData]) -> None:
    # Direct import, no forward references needed
    pass

# app/features/generators/schemas.py
from app.schemas.lora import LoRAData

class GeneratorConfig(BaseModel):
    loras: list[LoRAData] = Field(default_factory=list)
```

**Benefits:**
- No TYPE_CHECKING workarounds needed
- Clear dependency structure
- Easier to maintain and refactor
- Better IDE support and type checking

## Code Duplication and Modularity

**Extract duplicated code into shared utilities to maintain SonarQube quality gates (≤ 3% duplication):**

- Extract when same logic appears in 2+ places (especially 10+ lines)
- Extract when logic is likely to change together
- Create shared function in `app/cores/` or `app/services/`
- Place generation-related utilities in `app/cores/generation/`
- Place infrastructure utilities in `app/services/`
- Use shared schemas in `app/schemas/` to avoid circular dependencies
- Service methods can delegate to shared utilities
- Add comprehensive tests for shared utilities (aim for 80%+ coverage)

**Modularity principles:**

- Split large files into focused modules with single responsibilities
- Keep main service files as thin orchestration layers
- Group related functions into separate files by responsibility
- Use clear, descriptive filenames that indicate purpose
