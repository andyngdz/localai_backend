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
    return file_path.rsplit('/', 1)[0]  # Hard to read, error-prone

def get_filename_from_path(file_path: str) -> str:
    return file_path.split('/')[-1]  # What does [-1] mean?
```

**Good example (using pathlib):**

```python
from pathlib import PurePosixPath

def get_directory_from_path(file_path: str) -> str:
    parent = str(PurePosixPath(file_path).parent)
    return parent if parent != '.' else ''

def get_filename_from_path(file_path: str) -> str:
    return PurePosixPath(file_path).name
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

## Type Safety

Fix type errors at their source—never use `# type: ignore` to bypass warnings. When mypy reports an error:

- Define proper types (TypedDict, Pydantic models, Protocol)
- Use `cast()` with explanatory comments for legitimate type narrowing
- Add type annotations to function signatures when library stubs are incomplete

**Never use `any` type.** It defeats the purpose of type checking. Instead:

- Use specific types or Union types
- Use TypeVar for generic types
- Use Protocol for duck-typed interfaces
- Use proper type annotations even if it requires more work

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
        >>> get_directory_from_path('a/b/c/model.bin')
        'a/b/c'
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
