# Repository Guidelines

## Project Structure & Module Organization
The FastAPI entrypoint is `main.py`, with shared settings in `config.py`. Feature routers live under `app/features/...`, supported by services in `app/cores` and `app/services`, persistence code in `app/database`, Socket.IO glue in `app/socket`, and shared Pydantic models in `app/schemas`. Alembic migration scripts reside in `alembic/`. Tests mirror the feature layout inside `tests/app`. Generated media and other static assets belong in `static/`; keep that directory light in commits.

## Build, Test, and Development Commands
Install dependencies with `uv sync` (or `pip install -r requirements.txt` when uv is unavailable). Apply migrations via `alembic upgrade head` before hitting the API. Run the dev server with `uvicorn main:app --reload` for live reloads; `uvicorn main:app --host 0.0.0.0 --port 8000` matches production defaults. Use `ruff check` to lint, `ruff format` to auto-format, and `pytest -q --cov=app --cov-report=xml` to execute the suite and refresh `coverage.xml`.

## Coding Style & Naming Conventions
Let Ruff format Python modules—tabs for indentation, 120-character lines, and single-quoted strings are enforced. Follow standard FastAPI patterns: async route handlers where possible, `snake_case` for functions, `PascalCase` for classes, and module-level constants in `ALL_CAPS`. Prefer explicit type hints on public interfaces; optional mypy checks (`uv run mypy`) should stay clean.

## Testing Guidelines
Write tests with pytest inside `tests/app`, mirroring the module under test. Name files `test_<feature>.py` and isolate concerns with fixtures from `tests/conftest.py`. Aim to cover new routes and services with both success and failure paths, keeping coverage from regressing when `coverage.xml` is regenerated.

## Commit & Pull Request Guidelines
Existing history follows Conventional Commit prefixes (`fix:`, `refactor:`, `feat:`). Keep commit bodies focused on the “why” and reference ticket IDs when relevant. Pull requests should: summarise the change, link related issues, note migrations or model downloads, and include the commands you ran (tests, lint, server). Screenshot UI or API contract changes when they affect external consumers.
