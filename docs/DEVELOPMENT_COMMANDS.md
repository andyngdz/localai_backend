# Development Commands

**Running the application:**

```bash
uv run python main.py  # or: uv run uvicorn main:app --reload
```

**Testing:**

```bash
uv run pytest                    # Run all tests
uv run pytest -q                 # Quiet mode
uv run pytest tests/path/to/file.py  # Specific file
uv run pytest -q --cov=app       # With coverage
```

**Code quality:**

```bash
uv run ruff format               # Format (tabs, single quotes)
uv run ruff check                # Lint
uv run ruff check --fix          # Lint and fix
uv run mypy app tests            # Type checking
```

**Database:** Standard Alembic migrations (`uv run alembic upgrade head`, `uv run alembic revision --autogenerate -m "msg"`)
