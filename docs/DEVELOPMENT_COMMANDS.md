# Development Commands

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
