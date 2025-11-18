# Error Handling

**Exceptions:** Create domain-specific exceptions with meaningful names (`ModelLoadCancelledException`) and include context (ids, reasons).

**Never use bare `except` statements.** Always specify the exception type to catch. Bare except catches system exits and keyboard interrupts, which should not be suppressed.

**Bad example:**

```python
try:
    await service.generate_image(sample_config, mock_db)
except:  # E722: Do not use bare 'except'
    pass
```

**Good example:**

```python
try:
    await service.generate_image(sample_config, mock_db)
except Exception:  # Specify exception type
    pass

# Or better - catch specific exceptions:
try:
    await service.generate_image(sample_config, mock_db)
except ValueError as error:
    logger.error(f"Generation failed: {error}")
except torch.cuda.OutOfMemoryError:
    logger.error("Out of memory")
```

**HTTP status codes:**

- `200` Success | `400` Bad input | `404` Not found | `409` Conflict | `500` Unexpected failure

**Responses:** Use Pydantic schemas (never raw dicts), include status/reason fields, provide context in error messages.
