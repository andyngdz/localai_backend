# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Switch to .venv

```bash
source .venv/bin/activate
```

### Running the Application

```bash
python main.py
# or
uvicorn main:app --reload
```

### Testing

```bash
# Run all tests with coverage
pytest -q --cov=app --cov-report=xml:coverage.xml --cov-report=term

# Run tests with basic output
pytest

# Run specific test file
pytest tests/app/features/downloads/test_api.py

# Run with verbose output
pytest -v
```

### Code Quality

```bash
# Format code (uses single quotes, tabs for indentation)
ruff format

# Lint code
ruff check

# Lint and fix issues
ruff check --fix
```

### Database Operations

```bash
# Database migrations are handled via Alembic
# Generate migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Downgrade migration
alembic downgrade -1
```

### Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install both
pip install -r requirements.txt -r requirements-dev.txt
```

## Architecture Overview

### Application Structure

- **FastAPI Application**: Main entry point in `main.py` with lifespan management
- **Feature-based Architecture**: Code organized by features under `app/features/`
- **Core Services**: Shared business logic in `app/cores/`
- **Database Layer**: SQLAlchemy models and CRUD operations in `app/database/`
- **Service Layer**: Business logic services in `app/services/` (models, storage, etc.)
- **WebSocket Support**: Real-time communication via Socket.IO

### Key Components

**Features** (`app/features/`):

- `generators/` - AI image generation endpoints and logic
- `downloads/` - Model download management
- `hardware/` - System hardware information
- `histories/` - Generation history tracking
- `models/` - Model management APIs
- `styles/` - Predefined image styles
- `users/` - User management
- `resizes/` - Image resizing functionality

**Core Services** (`app/cores/`):

- `model_manager/` - AI model loading and management
- `model_loader/` - Model loading utilities with device optimization
- `samplers/` - Sampling algorithms and schedulers
- `constants/` - Shared constants and configurations

**Database Models** (`app/database/models/`):

- `Model` - AI model metadata (model_id, model_dir)
- `GeneratedImage` - Generated image records
- `History` - Generation history entries
- `Config` - Application configuration

### Service Layer

- `app/services/` contains utility services:
  - `device.py` - GPU/CPU device management
  - `image.py` - Image processing utilities
  - `storage.py` - File storage management
  - `logger.py` - Logging configuration
  - `memory.py` - Memory usage monitoring

### Configuration

- Database: SQLite (`localai_backend.db`)
- Static files served from `static/` directory
- Generated images stored in `static/generated_images/`
- Cache directory: `.cache/`
- CORS enabled for all origins

### Development Notes

- Uses tab indentation and single quotes (configured in `ruff.toml`)
- Async/await patterns throughout FastAPI endpoints
- SQLAlchemy 2.0 with declarative base
- Comprehensive test coverage with pytest and pytest-asyncio
- CI/CD pipeline includes code quality checks and SonarCloud analysis
- **Always use Pydantic schemas for JSON responses, never return raw dict objects**

## Common Mistakes to Avoid

**1. Never use underscore prefixes for methods/attributes**

- Use public names by default: `lock` not `_lock`, `set_state()` not `_set_state()`
- Only use underscores for truly private implementation details
- The codebase preference is explicit public interfaces

**2. Question every code pattern before copying it**

- Don't cargo cult program (copy patterns without understanding why they exist)
- Before adding code, ask:
  - "What problem does this solve?"
  - "Does this apply to my use case?"
  - "What's the performance impact?"
- Test with real workloads, not just unit tests

**3. Test with real-world loads before claiming success**

- Unit tests passing ≠ code is performant or correct
- Check application logs for warnings and timing issues
- Verify the solution doesn't introduce worse problems than it solves

## Communication Guidelines

**ALWAYS ask clarifying questions before proposing solutions. NEVER make assumptions about user intent.**

- Use the `AskUserQuestion` tool to present questions as selectable options/checkboxes
- Maximum 4 options per question (tool limitation)
- Each option needs: `label` (short title), `description` (explanation)
- Get full context before jumping into implementation

**When to ask questions:**

- Implementation tasks (new features, refactoring, bug fixes)
- Ambiguous requests: "optimize this", "fix the error", "improve performance"
- Complex requests: "add authentication", "integrate with API"
- Continuation scenarios: "continue", "keep going" → Ask which specific task
- Any request where multiple valid approaches exist

**When NOT to ask (trivial commands):**

- Simple test runs: "run tests", "pytest"
- Code formatting: "format code", "ruff format"
- Informational: "show me the logs", "what does this function do"
- Explicit with clear intent: "run tests on file X", "read function Y at line Z"

**Examples:**

- User: "optimize this" → Ask: speed, memory, readability, or code size?
- User: "fix the bug" → Ask: which bug, where, what's expected behavior?
- User: "continue" → Ask: continue which task (list recent incomplete tasks)?
- User: "add auth" → Ask: method (JWT/OAuth), token storage, session duration?

## Testing Strategy

**When to write tests:**

- Always write tests for new features or bug fixes
- Add tests before marking implementation as complete
- Update existing tests when changing behavior

**Test organization:**

- Place tests in `tests/` directory mirroring `app/` structure
- Use descriptive test class names: `TestFeatureName` format
- Group related tests in classes (e.g., `TestLoadModelEndpoint`)
- Each test method should test one specific behavior

**Coverage expectations:**

- Aim for high coverage on core business logic
- Test happy paths and error cases
- Include edge cases (cancellation, timeouts, race conditions)
- Run tests with coverage: `pytest -q --cov=app`

**Test patterns:**

- Use `pytest.mark.asyncio` for async tests
- Mock external dependencies (database, network, GPU operations)
- Use fixtures for common test setup
- Verify both behavior and side effects (state changes, logs)

## Error Handling Patterns

**Exception hierarchy:**

- Create custom exceptions for domain-specific errors
- Use meaningful exception names: `ModelLoadCancelledException`, not `Error`
- Include context in exceptions (reason, ids, etc.)

**HTTP status codes:**

- `200 OK`: Success (including expected cancellations with status field)
- `400 Bad Request`: Invalid input, validation errors
- `404 Not Found`: Resource doesn't exist
- `409 Conflict`: Resource in use, state conflict
- `500 Internal Server Error`: Unexpected failures only

**Logging levels:**

- `logger.info()`: Expected operations (cancellations, state changes)
- `logger.warning()`: Recoverable issues, degraded performance
- `logger.error()`: Unexpected failures, exceptions
- `logger.exception()`: Errors with full stack trace

**Response patterns:**

- Always return Pydantic schemas, never raw dicts
- Include status/reason fields for non-standard responses
- Provide helpful error messages with context
