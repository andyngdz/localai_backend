# Testing

Write tests for all new features and bug fixes before marking work complete. Tests should mirror `app/` structure in `tests/` directory and use descriptive class names (`TestFeatureName`).

**Test patterns:**

- Use `pytest.mark.asyncio` for async tests
- Mock external dependencies (database, GPU, network)
- Cover happy paths, error cases, and edge cases (cancellation, timeouts, races)
- Verify behavior and side effects (state changes, logs)
