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
- Type hints required (mypy enforced)

**Pre-commit validation:** Husky runs `uv run ruff format --check`, `uv run ruff check`, `uv run mypy`, and `uv run pytest -q` on every commit. Commits are blocked if any check fails.
