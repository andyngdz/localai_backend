# Coding Style

Fix type errors at their source—never use `# type: ignore` to bypass warnings. When mypy reports an error:

- Define proper types (TypedDict, Pydantic models, Protocol)
- Use `cast()` with explanatory comments for legitimate type narrowing
- Add type annotations to function signatures when library stubs are incomplete

Use public interfaces by default (`lock`, `set_state()`) and reserve underscores for truly private implementation details.
