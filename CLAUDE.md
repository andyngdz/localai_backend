# Claude Code Guide

## Essential Documentation

Read these files before working on tasks:

- **Architecture**: @docs/ARCHITECTURE.md - System design, stack, circular imports, modularity
- **Coding Style**: @docs/CODING_STYLE.md - Code standards with skill references
- **Development Commands**: @docs/DEVELOPMENT_COMMANDS.md - Essential commands

## Logging

Use `LoggerService` with mandatory categories. See `app/services/logger.py`.

**Usage:**

```python
from app.services import logger_service
logger = logger_service.get_logger(__name__, category='ModelLoad')
logger.info('Loading model...')
```

**Categories:** `[ModelLoad]` `[Download]` `[Generate]` `[API]` `[Database]` `[Service]` `[Socket]` `[GPU]`

**Levels:** `.debug()` `.info()` `.warning()` `.error()` `.exception()`

**Config:** `LOG_LEVEL=DEBUG uv run python main.py`

## Error Handling

- **Never use bare `except`** - always specify exception type
- Create domain-specific exceptions with meaningful names
- Use Pydantic schemas for responses (never raw dicts)
- **HTTP codes:** 200 Success | 400 Bad input | 404 Not found | 409 Conflict | 500 Failure

**Good example:**

```python
try:
    await service.generate_image(config, db)
except ValueError as error:
    logger.error(f"Generation failed: {error}")
except torch.cuda.OutOfMemoryError:
    logger.error("Out of memory")
```

## Testing

Write tests for all features/fixes before marking complete. Mirror `app/` structure in `tests/`.

**Patterns:**
- Use `pytest.mark.asyncio` for async tests
- Mock external dependencies (database, GPU, network)
- Cover happy paths, error cases, edge cases

**Before finishing, run:**

```bash
uv run pyright app/path/to/modified.py tests/path/to/test.py
uv run ruff check --fix app/ tests/
uv run ruff format app/ tests/
uv run pytest -q
```

## Communication

Ask clarifying questions for ambiguous/complex requests:
- **Ambiguous:** "optimize this" → Ask: speed, memory, or readability?
- **Complex:** "add authentication" → Ask: JWT, OAuth, session duration?

Skip questions for trivial commands like "run tests" or "format code".

Never create/amend commits without explicit permission.

## Plan Mode

When exiting plan mode, save the implementation plan in `plans/{serial-number}-{plan-name}.md`

## Available Skills

For detailed patterns, use these skills:

- `type-safety-mastery` - Type safety rules, Pydantic models, type stubs
- `refactoring-patterns` - Extract variable, split temp, replace temp with query
- `code-organization` - Encapsulation, modularity, function design, naming
