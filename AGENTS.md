# Repository Guidelines

## Project Structure & Module Organization

- Entry point: `main.py` (FastAPI app, lifespan, CORS, static, Socket.IO).
- Source: `app/`
  - `features/` (routers per domain: `downloads/`, `generators/`, `hardware/`, `histories/`, `models/`, `resizes/`, `styles/`, `users/`)
  - `cores/` (model loading/management, samplers, constants)
  - `database/` (SQLAlchemy + Alembic integration)
  - `services/` (device, image, logger, storage, etc.)
  - `socket/` (Socket.IO integration)
  - `schemas/` (Pydantic models)
- Migrations: `alembic/` and `alembic.ini`
- Static assets: `static/` (e.g., `static/generated_images/`)
- Tests: `tests/` mirroring `app/` layout.

## Build, Test, and Development Commands

- Setup env: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt -r requirements-dev.txt`
- Run app: `uvicorn main:app --reload` (or `python main.py`)
- Lint/format: `ruff check` | `ruff check --fix` | `ruff format`
- Tests: `pytest -q --cov=app --cov-report=xml:coverage.xml`
- Alembic: `alembic revision --autogenerate -m "msg"` then `alembic upgrade head`

## Coding Style & Naming Conventions

- Python 3.11+. Tabs for indentation (indent-width 2), single quotes, line length 120 (see `ruff.toml`).
- Names: modules/files `snake_case.py`; functions/vars `snake_case`; classes `PascalCase`; constants `UPPER_CASE`.
- Prefer type hints and FastAPI `Response` models; keep endpoints async.
- Keep feature code inside `app/features/<feature>/` with a `router`.

## Testing Guidelines

- Frameworks: `pytest`, `pytest-asyncio`, `pytest-cov`.
- Structure tests to mirror modules, e.g., `tests/app/features/models/test_api.py`.
- Naming: files start with `test_*.py`; use markers `@pytest.mark.slow`/`integration` when relevant.
- Aim for meaningful coverage; `coverage.xml` is generated for CI tools.

## Commit & Pull Request Guidelines

- Follow Conventional Commits: `feat:`, `fix:`, `refactor:`, `chore:`, `test:`, `ci:` (see `git log`).
- One logical change per commit; clear subject, brief body if needed.
- PRs must: describe changes, link issues, include tests for new behavior, and pass lint/tests. Add screenshots or sample requests for API changes when helpful.

## Security & Configuration Tips

- Do not commit secrets. Defaults live in `config.py`; prefer env vars for overrides when adding new config.
- Database: SQLite by default; manage schema via Alembic migrations.
- Large artifacts go under `static/` or `.cache/` and should be ignored by Git where appropriate.
