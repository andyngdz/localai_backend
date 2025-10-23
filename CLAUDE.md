# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Development Commands

**Running the application:**
```bash
python main.py  # or: uvicorn main:app --reload
```

**Testing:**
```bash
pytest                    # Run all tests
pytest -q                 # Quiet mode
pytest tests/path/to/file.py  # Specific file
pytest -q --cov=app       # With coverage
```

**Code quality:**
```bash
ruff format               # Format (tabs, single quotes)
ruff check                # Lint
ruff check --fix          # Lint and fix
mypy app tests            # Type checking
```

**Database:** Standard Alembic migrations (`alembic upgrade head`, `alembic revision --autogenerate -m "msg"`)

## Project Architecture

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
- Type hints required (mypy enforced)

**Pre-commit validation:** Husky runs `ruff format --check`, `ruff check`, `mypy`, and `pytest -q` on every commit. Commits are blocked if any check fails.

## Engineering Principles

### 1. Type Safety & Code Quality

Fix type errors at their source—never use `# type: ignore` to bypass warnings. When mypy reports an error:
- Define proper types (TypedDict, Pydantic models, Protocol)
- Use `cast()` with explanatory comments for legitimate type narrowing
- Add type annotations to function signatures when library stubs are incomplete

Use public interfaces by default (`lock`, `set_state()`) and reserve underscores for truly private implementation details.

### 2. Validation & Real-World Testing

Understand code patterns before reusing them—ask "What problem does this solve?" and "Does this apply here?" Unit tests passing doesn't guarantee correctness or performance:
- Test with realistic workloads
- Check logs for warnings and timing issues
- Verify fixes don't introduce new problems

### 3. Communication & Workflow

Ask clarifying questions for ambiguous or complex requests using the `AskUserQuestion` tool (max 4 options per question). Examples:
- **Ambiguous:** "optimize this" → Ask: speed, memory, or readability?
- **Complex:** "add authentication" → Ask: JWT, OAuth, session duration?
- **Continuation:** "continue" → Ask: which specific task?

Skip questions for trivial commands like "run tests" or "format code".

Never create or amend commits without explicit permission—the user manages git operations.

## Development Practices

### Testing

Write tests for all new features and bug fixes before marking work complete. Tests should mirror `app/` structure in `tests/` directory and use descriptive class names (`TestFeatureName`).

**Test patterns:**
- Use `pytest.mark.asyncio` for async tests
- Mock external dependencies (database, GPU, network)
- Cover happy paths, error cases, and edge cases (cancellation, timeouts, races)
- Verify behavior and side effects (state changes, logs)

### Error Handling

**Exceptions:** Create domain-specific exceptions with meaningful names (`ModelLoadCancelledException`) and include context (ids, reasons).

**HTTP status codes:**
- `200` Success | `400` Bad input | `404` Not found | `409` Conflict | `500` Unexpected failure

**Logging:**
- `.info()` Expected operations | `.warning()` Recoverable issues | `.error()` Failures | `.exception()` With stack trace

**Responses:** Use Pydantic schemas (never raw dicts), include status/reason fields, provide context in error messages.
