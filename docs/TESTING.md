# Testing

Write tests for all new features and bug fixes before marking work complete. Tests should mirror `app/` structure in `tests/` directory and use descriptive class names (`TestFeatureName`).

**Test patterns:**

- Use `pytest.mark.asyncio` for async tests
- Mock external dependencies (database, GPU, network)
- Cover happy paths, error cases, and edge cases (cancellation, timeouts, races)
- Verify behavior and side effects (state changes, logs)

**IMPORTANT: Always verify code quality before finishing tests:**

After writing or modifying tests, you MUST run these checks before considering the work complete:

1. **Check type errors** (pyright on modified files):
   ```bash
   uv run pyright app/path/to/modified.py tests/path/to/test.py
   ```

2. **Fix linting issues**:
   ```bash
   uv run ruff check --fix app/ tests/
   ```

3. **Format code**:
   ```bash
   uv run ruff format app/ tests/
   ```

4. **Run the test suite** to ensure all tests pass:
   ```bash
   uv run pytest -q
   ```

Never mark work as complete without running all four steps above. This ensures:
- No type errors (pyright)
- No linting errors (ruff)
- Consistent formatting (ruff)
- All tests pass (pytest)
