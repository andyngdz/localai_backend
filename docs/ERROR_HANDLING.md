# Error Handling

**Exceptions:** Create domain-specific exceptions with meaningful names (`ModelLoadCancelledException`) and include context (ids, reasons).

**HTTP status codes:**

- `200` Success | `400` Bad input | `404` Not found | `409` Conflict | `500` Unexpected failure

**Responses:** Use Pydantic schemas (never raw dicts), include status/reason fields, provide context in error messages.
