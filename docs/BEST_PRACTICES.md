# Best Practices

Understand code patterns before reusing them—ask "What problem does this solve?" and "Does this apply here?" Unit tests passing doesn't guarantee correctness or performance:

- Test with realistic workloads
- Check logs for warnings and timing issues
- Verify fixes don't introduce new problems

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

## Import Patterns

**Use consistent import aliases for service modules.** The codebase uses `database_service` as the standard alias for CRUD operations.

**Bad example:**

```python
from app.database.crud import get_lora_by_id, add_lora, delete_lora

lora = get_lora_by_id(db, lora_id)
```

**Good example:**

```python
from app.database import crud as database_service

lora = database_service.get_lora_by_id(db, lora_id)
```

**Benefits:**
- Consistent pattern across the codebase
- Clear namespace separation
- Easier to mock in tests
- Better code readability

## Code Clarity in Loops

**Extract reused object fields into variables.** When accessing the same field multiple times, extract it once for clarity and maintainability.

**Bad example:**

```python
for config in lora_configs:
    logger.info(f"Loading LoRA '{config.name}' (weight: {config.weight})")
    try:
        self.pipe.load_lora_weights(config.file_path, adapter_name=adapter_name)
    except Exception as error:
        logger.error(f"Failed to load LoRA '{config.name}': {error}")
        raise ValueError(f"Failed to load LoRA '{config.name}': {error}")
```

**Good example:**

```python
for config in lora_configs:
    name = config.name  # Extract once, reuse

    logger.info(f"Loading LoRA '{name}' (weight: {config.weight})")
    try:
        self.pipe.load_lora_weights(config.file_path, adapter_name=adapter_name)
    except Exception as error:
        logger.error(f"Failed to load LoRA '{name}': {error}")
        raise ValueError(f"Failed to load LoRA '{name}': {error}")
```

**Benefits:**
- Reduces repetition
- Easier to refactor if field access changes
- More performant (single attribute lookup)
- Clearer intent

## Using Unique Identifiers

**Use database IDs or unique identifiers for adapter/instance names.** When working with multiple instances of objects (adapters, plugins, etc.), use database IDs or UUIDs for guaranteed uniqueness.

**Bad example:**

```python
for idx, config in enumerate(lora_configs):
    adapter_name = f"lora_{idx}"  # Index can change if list order changes
    self.pipe.load_lora_weights(config.file_path, adapter_name=adapter_name)
```

**Good example:**

```python
for config in lora_configs:
    adapter_name = f"lora_{config.id}"  # Database ID is stable and unique
    self.pipe.load_lora_weights(config.file_path, adapter_name=adapter_name)
```

**Benefits:**
- Guaranteed uniqueness
- Stable across reorderings
- Easier to debug (can trace back to database)
- More predictable behavior

## Code Duplication

**Extract duplicated code into shared utilities to maintain SonarQube quality gates (≤ 3% duplication):**

- Extract when same logic appears in 2+ places (especially 10+ lines)
- Extract when logic is likely to change together
- Create shared function in `app/cores/` or `app/services/`
- Place generation-related utilities in `app/cores/generation/`
- Place infrastructure utilities in `app/services/`
- Use TYPE_CHECKING and local imports to avoid circular dependencies
- Service methods can delegate to shared utilities
- Add comprehensive tests for shared utilities (aim for 80%+ coverage)
