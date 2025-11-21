# Coding Style

## Documentation First

**Always check documentation before implementing code.** Use MCP tools or WebSearch to verify:

- Latest API patterns and best practices
- Correct usage of libraries and frameworks
- Security considerations and gotchas
- Performance optimizations

Never guess at API usage—look it up first.

## Use Standard Library Solutions

**Always check if Python's standard library has a solution before writing custom code.** Common tasks like path manipulation, JSON parsing, HTTP requests, etc. usually have built-in solutions.

**Research standard library first:**

- Check `pathlib` for path operations
- Check `os.path` for file system operations
- Check `json`, `csv`, `configparser` for data formats
- Check `urllib`, `http.client` for network operations
- Check `datetime`, `time` for time operations

**Bad example (custom string manipulation):**

```python
def get_directory_from_path(file_path: str) -> str:
    if '/' not in file_path:
        return ''
    return file_path.rsplit('/', 1)[0]
```

**Good example (using pathlib):**

```python
from pathlib import PurePosixPath

def get_directory_from_path(file_path: str) -> str:
    parent = str(PurePosixPath(file_path).parent)
    return parent if parent != '.' else ''
```

**Benefits:**

- Self-documenting and clear intent
- Type-safe and tested by Python core team
- Cross-platform and handles edge cases
- Follows Python idioms and best practices

**If unsure whether a standard library solution exists:**

1. Use MCP or WebSearch to research
2. Ask "Is there a Python standard library for X?"
3. Check the Python docs

## Functional Programming with Pydash

**Use `pydash` for list/collection operations when it improves readability.** Pydash provides a rich set of functional programming helpers that can make complex list transformations and filtering cleaner than nested list comprehensions.

**Bad example (nested list comprehension):**

```python
def filter_files(files, scopes):
    return [
        file_path for file_path in files
        if any(fnmatch.fnmatch(file_path, scope) for scope in scopes)
    ]
```

**Good example (using pydash):**

```python
import pydash

def filter_files(files, scopes):
    return list(
        pydash.filter_(
            files,
            lambda file_path: pydash.some(scopes, lambda scope: fnmatch.fnmatch(file_path, scope))
        )
    )
```

**When to use pydash:**

- Complex filtering logic involving multiple conditions
- Chaining multiple operations (map, filter, reduce)
- Working with deeply nested data structures (using `pydash.get`)
- When list comprehensions become hard to read (more than 2 lines or nested)

## Type Safety

Fix type errors at their source—never use `# type: ignore` to bypass warnings. When pyright reports an error:

- Define proper types (Pydantic models, Protocol)
- Add type annotations to function signatures when library stubs are incomplete

**Never use `# type: ignore[return-value]` or any specific type ignore comments.** If a function's return type doesn't match:

- Fix the actual return type
- Use proper type annotations
- Refactor the code to match the declared type

**Never use `any` type.** It defeats the purpose of type checking. Instead:

- Use specific types or Union types
- Use TypeVar for generic types
- Use Protocol for duck-typed interfaces
- Use proper type annotations even if it requires more work

**Prefer `Optional[T]` over `T | None`:**

- Use `Optional[str]` instead of `str | None`
- It is more explicit and readable
- Consistent with the codebase style

**Minimize use of `cast` and runtime workarounds:**

- Avoid using `cast(Type, val)` to silence type errors when possible
- Do not use `getattr(obj, 'attr')` or `setattr(obj, 'attr', val)` to bypass missing type definitions
- **Prefer fixing the root cause:** Update the type stubs in `typings/` instead
- When `cast` is unavoidable (for example, dynamically resolved methods), add a short comment explaining why

**Use Pydantic models instead of TypedDict for data structures:**

- Pydantic provides runtime validation
- Better consistency with the rest of the codebase
- Clear error messages when validation fails
- Better IDE support and autocomplete

**Always prefer `BaseModel` over `Dict`/`TypedDict`.** If you catch yourself annotating a variable as `dict[str, X]` (or using `TypedDict`) inside application code, extract a dedicated Pydantic model instead and expose helper methods (`get_size()`, `set_size()`, etc.) so callers never need `.get()` accessors.

**Bad example:**

```python
from typing import TypedDict

class LoRAData(TypedDict):
    id: int
    name: str
    weight: float
```

**Good example:**

```python
from pydantic import BaseModel, Field

class LoRAData(BaseModel):
    id: int = Field(..., description='Database ID')
    name: str = Field(..., description='Display name')
    weight: float = Field(..., ge=0.0, le=2.0, description='Weight/strength')
```

Use public interfaces by default (`lock`, `set_state()`) and reserve underscores for truly private implementation details.

**Type aliases for complex types:** Create type aliases for frequently used complex types to improve readability.

**Bad example:**

```python
def list_files(
    self, id: str,
    repo_info: Optional[Union[ModelInfo, DatasetInfo, SpaceInfo]] = None
) -> List[str]:
    pass

def get_file_sizes(
    self, id: str,
    repo_info: Optional[Union[ModelInfo, DatasetInfo, SpaceInfo]] = None
) -> Dict[str, int]:
    pass
```

**Good example:**

```python
# Define type alias once at module level
RepoInfo = Union[ModelInfo, DatasetInfo, SpaceInfo]

def list_files(self, id: str, repo_info: Optional[RepoInfo] = None) -> List[str]:
    pass

def get_file_sizes(self, id: str, repo_info: Optional[RepoInfo] = None) -> Dict[str, int]:
    pass
```

**Use type stubs (.pyi files) for external library types:**

- Create stub files in `typings/{package_name}/` instead of runtime wrapper classes
- **Update existing stubs** when you encounter missing attributes or methods
- Stubs provide type hints without runtime overhead
- Follow PEP 561 conventions (`.pyi` extension)
- Configure `stubPath = "typings"` in `pyproject.toml` under `[tool.pyright]`
- Never use runtime assertions (`assert isinstance(...)`) to force types
- Never use `TYPE_CHECKING`. If you have to use it, then you did it wrong. Go back the find the root cause, then fix it

## Documentation

**Add examples to docstrings for helper functions.** Examples make the function's behavior immediately clear and serve as inline tests.

**Bad example:**

```python
def get_directory_from_path(file_path: str) -> str:
    """Extract directory path from a file path."""
    parent = str(PurePosixPath(file_path).parent)
    return parent if parent != '.' else ''
```

**Good example:**

```python
def get_directory_from_path(file_path: str) -> str:
    """Extract directory path from a file path.

    Examples:
        >>> get_directory_from_path('unet/model.safetensors')
        'unet'
        >>> get_directory_from_path('model.safetensors')
        ''
    """
    parent = str(PurePosixPath(file_path).parent)
    return parent if parent != '.' else ''
```

**When to add examples:**

- Helper functions with non-obvious behavior
- Functions that handle edge cases (empty strings, None values)
- Path manipulation, string parsing, data transformation functions
- Public API methods

## Encapsulation

**Use wrapper methods instead of exposing internal dependencies.** This provides better encapsulation and makes refactoring easier.

**Bad example:**

```python
class DownloadService:
    def __init__(self):
        self.repository = HuggingFaceRepository()

    def download_model(self, id: str):
        # Directly accessing internal API of repository
        repo_info = self.repository.api.repo_info(id)
        ...
```

**Good example:**

```python
class HuggingFaceRepository:
    def get_repo_info(self, id: str) -> RepoInfo:
        """Get repository information from HuggingFace Hub."""
        return self.api.repo_info(id)

class DownloadService:
    def __init__(self):
        self.repository = HuggingFaceRepository()

    def download_model(self, id: str):
        # Using wrapper method - better encapsulation
        repo_info = self.repository.get_repo_info(id)
        ...
```

**Benefits:**

- Hides implementation details
- Makes it easier to add logging, caching, or error handling
- Simplifies testing (mock the wrapper instead of internal dependencies)
- Allows changing the underlying implementation without affecting callers

## File Modularity

**Never put everything in one file.** Split large files into focused modules with single responsibilities.

**When to split:**

- File exceeds 150 lines
- Class has more than 5 distinct responsibilities
- Logic can be grouped into clear, reusable modules

**How to split:**

- Group related functions into separate files
- Create modules by responsibility (e.g., `repository.py`, `filters.py`, `file_downloader.py`)
- Keep main service file as a thin orchestration layer
- Use clear, descriptive filenames that indicate purpose

**Example structure:**

```
features/downloads/
  ├── services.py          # Main orchestration only
  ├── repository.py        # Repository operations
  ├── file_downloader.py   # Low-level file operations
  └── filters.py           # Filtering logic
```

## Function Design

**One function, one responsibility.** Each function should do exactly one thing and do it well.

**Bad example:**

```python
def process_user(user_data):
    # Validates, saves, and sends email - too many responsibilities
    if not user_data.get('email'):
        raise ValueError('Email required')
    db.save(user_data)
    send_welcome_email(user_data['email'])
    return user_data
```

**Good example:**

```python
def validate_user_data(user_data):
    if not user_data.get('email'):
        raise ValueError('Email required')

def save_user(user_data):
    return db.save(user_data)

def send_welcome_email(email):
    # Only handles email sending
    ...
```

## Code Clarity

**Use descriptive variable names in loops.** Never use single letters that don't convey meaning.

**Bad examples:**

```python
for p in components_scopes:  # What is 'p'?
for i in users:              # 'i' suggests index, but it's a user
for x in files:              # Meaningless
```

**Good examples:**

```python
for component_scope in components_scopes:
for scope in components_scopes:  # If 'scope' is clear in context
for user in users:
for file_path in files:
```

**Minimize comments—write self-documenting code instead.**

- Only add comments for non-obvious business logic or workarounds
- Never comment on what the code does (code should be clear enough)
- Only comment on why it does it (when it's not obvious)
- Add comments for hacky solutions that can't be avoided

**Bad examples:**

```python
# Increment counter
counter += 1

# Loop through users
for user in users:
    # Process user
    process_user(user)
```

**Good examples:**

```python
# Workaround for HuggingFace API returning inconsistent revision formats
# See: https://github.com/huggingface/huggingface_hub/issues/1234
revision = getattr(repo_info, 'sha', None) or 'main'

# Skip lock acquisition here because this method is always called
# within a context that already holds the lock
self._update_state_unsafe(new_state)
```

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

## Clean Code Principles

**Never use hacky solutions.** Write code that is:

- Clear and maintainable
- Uses proper APIs and patterns
- Follows language idioms
- Can be understood by others

**If you find yourself writing a hack:**

1. Search for the proper solution in documentation
2. Refactor to use correct patterns
3. If truly unavoidable, document why with a comment
4. Create a TODO to fix it properly later

**Remember:** Code is read more often than written. Prioritize clarity over cleverness.

**Understand code patterns before reusing them:** Ask "What problem does this solve?" and "Does this apply here?" Unit tests passing doesn't guarantee correctness or performance:

- Test with realistic workloads
- Check logs for warnings and timing issues
- Verify fixes don't introduce new problems
