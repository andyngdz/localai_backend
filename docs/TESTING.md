# Testing

Write tests for all new features and bug fixes before marking work complete. Tests should mirror `app/` structure in `tests/` directory and use descriptive class names (`TestFeatureName`).

**Test patterns:**

- Use `pytest.mark.asyncio` for async tests
- Mock external dependencies (database, GPU, network)
- Cover happy paths, error cases, and edge cases (cancellation, timeouts, races)
- Verify behavior and side effects (state changes, logs)

**IMPORTANT: Always verify code quality before finishing tests:**

After writing or modifying tests, you MUST run these checks before considering the work complete:

1. **Run the test suite** to ensure all tests pass:
   ```bash
   uv run pytest tests/ -q
   ```

2. **Fix linting issues**:
   ```bash
   uv run ruff check --fix tests/
   ```

3. **Format code**:
   ```bash
   uv run ruff format tests/
   ```

4. **Verify tests still pass** after formatting:
   ```bash
   uv run pytest tests/ -q --tb=no
   ```

Never mark test work as complete without running all four steps above. This ensures:
- All tests pass
- Code follows project style guidelines
- No linting errors remain
- Formatting is consistent
