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

## Extract Variable Pattern

**Extract variables to eliminate repetition and improve code clarity.** This fundamental refactoring pattern (from Martin Fowler's "Refactoring") applies when expressions are repeated or when a descriptive name would clarify intent.

> "Any fool can write code that a computer can understand. Good programmers write code that humans can understand." - Martin Fowler

### When to Extract Variables:

1. **Repeated expressions** - Compute once, reuse multiple times
2. **Complex calculations** - Break down into named intermediate steps
3. **Unclear intent** - Use descriptive names to explain what a value represents
4. **Multiple field accesses** - Cache field lookups to improve performance

### Use Case 1: Extract Computed Values

Eliminate repetition of calculations and normalization logic.

**Bad example:**

```python
def set_size(self, filename: str, size: int) -> None:
    if filename in self._files_dict:
        self._files_dict[filename].size = max(size, 0)  # Repeated
    else:
        self._files_dict[filename] = RepositoryFileSize(
            filename=filename,
            size=max(size, 0)  # Repeated
        )
```

**Good example:**

```python
def set_size(self, filename: str, size: int) -> None:
    normalized_size = max(size, 0)  # Extract once, reuse

    if filename in self._files_dict:
        self._files_dict[filename].size = normalized_size
    else:
        self._files_dict[filename] = RepositoryFileSize(
            filename=filename,
            size=normalized_size
        )
```

### Use Case 2: Extract Complex Expressions

Use explaining variables to break down complex logic into understandable steps.

**Bad example:**

```python
if (user.is_active and user.subscription_end > datetime.now() and
    user.payment_status == 'paid' and user.role in ['premium', 'enterprise']):
    grant_access()
```

**Good example:**

```python
has_valid_subscription = user.subscription_end > datetime.now()
has_paid_status = user.payment_status == 'paid'
has_premium_role = user.role in ['premium', 'enterprise']
is_eligible = user.is_active and has_valid_subscription and has_paid_status and has_premium_role

if is_eligible:
    grant_access()
```

### Use Case 3: Extract Object Field Access

Cache repeated field lookups for clarity and performance.

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

### Use Case 4: Extract Function Call Results

Avoid redundant expensive operations (API calls, database queries, I/O).

**Bad example:**

```python
def download_model(self, model_id: str) -> None:
    logger.info(f"Downloading {self.repository.get_repo_info(model_id).modelId}")
    files = self.repository.list_files(model_id)

    if self.repository.get_repo_info(model_id).private:  # Redundant API call
        self._validate_token()
```

**Good example:**

```python
def download_model(self, model_id: str) -> None:
    repo_info = self.repository.get_repo_info(model_id)  # Call once

    logger.info(f"Downloading {repo_info.modelId}")
    files = self.repository.list_files(model_id)

    if repo_info.private:
        self._validate_token()
```

### When NOT to Extract:

1. **Single use** - Don't extract if used only once
2. **Obvious expressions** - `x + 1` doesn't need extraction
3. **Very short scope** - Within 2-3 adjacent lines where context is clear

**Over-extraction example (avoid):**

```python
# Too granular - reduces readability
one = 1
result = x + one  # Just use x + 1
```

### Related Patterns:

**Split Temporary Variable** - Use different variables for different purposes instead of reusing one variable.

**Bad example:**

```python
temp = base_price * quantity
logger.info(f"Subtotal: {temp}")

temp = temp * (1 + tax_rate)  # Reusing 'temp' for different purpose
logger.info(f"Total: {temp}")
```

**Good example:**

```python
subtotal = base_price * quantity
logger.info(f"Subtotal: {subtotal}")

total = subtotal * (1 + tax_rate)  # Clear purpose
logger.info(f"Total: {total}")
```

**Replace Temp with Query** - When a temporary variable can be replaced with a method call (useful for testability).

**Before:**

```python
base_price = quantity * item_price
if base_price > 1000:
    return base_price * 0.95
return base_price * 0.98
```

**After:**

```python
def base_price(self) -> float:
    return self.quantity * self.item_price

def final_price(self) -> float:
    if self.base_price() > 1000:
        return self.base_price() * 0.95
    return self.base_price() * 0.98
```

### Summary:

Extract variables when it makes code:
- **DRY** (Don't Repeat Yourself) - Compute once, use many times
- **Clear** - Descriptive names explain what values represent
- **Maintainable** - Changes happen in one place
- **Performant** - Avoid redundant expensive operations

**References:**
- Martin Fowler, "Refactoring: Improving the Design of Existing Code"
- Extract Variable (Introduce Explaining Variable)
- Split Temporary Variable
- Replace Temp with Query

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
