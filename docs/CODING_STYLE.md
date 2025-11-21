# Coding Style

**Before implementing code, always check documentation first.** Use MCP tools or WebSearch to verify latest API patterns, security considerations, and best practices. Never guess at API usage.

## Use Standard Library First

**Always check if Python's standard library has a solution before writing custom code.**

- `pathlib` for path operations
- `json`, `csv`, `configparser` for data formats
- `urllib`, `http.client` for network operations
- `datetime`, `time` for time operations

## Functional Programming with Pydash

**Use `pydash` for complex list/collection operations** when it improves readability over nested list comprehensions.

## Type Safety → Use type-safety-mastery skill

**Never use `# type: ignore` or `any` type.** Fix errors at source.

- Prefer `Optional[T]` over `T | None`
- Use Pydantic `BaseModel` over `Dict`/`TypedDict`
- Create type stubs in `typings/` for external libraries
- **Never use `TYPE_CHECKING`** - use shared schemas in `app/schemas/` instead

**For detailed type safety patterns, use the `type-safety-mastery` skill.**

## Documentation

**Add examples to docstrings for helper functions:**

```python
def get_directory_from_path(file_path: str) -> str:
    """Extract directory path from a file path.

    Examples:
        >>> get_directory_from_path('unet/model.safetensors')
        'unet'
    """
    parent = str(PurePosixPath(file_path).parent)
    return parent if parent != '.' else ''
```

## Code Organization → Use code-organization skill

- **Encapsulation**: Use wrapper methods instead of exposing internals
- **File Modularity**: Split files >150 lines
- **Function Design**: One function, one responsibility
- **Variable Naming**: Descriptive names in loops (never `i`, `x`, `p`)
- **Comments**: Only for non-obvious business logic, not what code does

**For detailed organization patterns, use the `code-organization` skill.**

## Refactoring → Use refactoring-patterns skill

**Extract variables to eliminate repetition:**

- Repeated expressions - compute once, reuse
- Complex calculations - break into named steps
- Multiple field accesses - cache lookups
- Function call results - avoid redundant operations

**For detailed refactoring patterns (Extract Variable, Split Temp, Replace Temp with Query), use the `refactoring-patterns` skill.**

## Import Patterns

**Use consistent import aliases:**

```python
from app.database import crud as database_service

lora = database_service.get_lora_by_id(db, lora_id)
```

## Clean Code Principles

**Never use hacky solutions.** Write code that is:

- Clear and maintainable
- Uses proper APIs and patterns
- Follows language idioms
- Can be understood by others

**Understand code patterns before reusing them.** Ask "What problem does this solve?" and "Does this apply here?"

Unit tests passing doesn't guarantee correctness or performance:
- Test with realistic workloads
- Check logs for warnings and timing issues
- Verify fixes don't introduce new problems

**Remember:** Code is read more often than written. Prioritize clarity over cleverness.
