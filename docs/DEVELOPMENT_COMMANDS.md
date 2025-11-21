# Development Commands

**Database migrations (run BEFORE starting the app):**

```bash
uv run alembic upgrade head                           # Apply pending migrations
uv run alembic revision --autogenerate -m "message"   # Create new migration
uv run alembic downgrade -1                           # Rollback last migration
```

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
uv run pyright app tests            # Type checking
uv run pyright path/to/file.py      # Type check specific file (faster)
```
