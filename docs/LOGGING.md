# Logging

The project uses a centralized `LoggerService` with colored console output and mandatory category support. See `app/services/logger.py`.

**Usage (category is mandatory):**

```python
from app.services import logger_service
logger = logger_service.get_logger(__name__, category='ModelLoad')
logger.info('Loading model...')  # Output: [INFO] ... [ModelLoad] Loading model...
```

**Standard categories:**

- `[ModelLoad]` Model loading, state management, resource cleanup, recommendations, pipeline conversion (8 files)
- `[Download]` Model download operations (2 files)
- `[Generate]` Image generation (txt2img, img2img), memory management, progress (7 files)
- `[API]` API endpoint requests/responses (5 files)
- `[Database]` Database operations (CRUD, config) (3 files)
- `[Service]` Infrastructure services (storage, device, platform, styles) (5 files)
- `[Socket]` WebSocket communication (1 file)
- `[GPU]` GPU memory management and utilities (1 file)

**Log levels:**

- `.debug()` Detailed diagnostic information
- `.info()` Expected operations and milestones
- `.warning()` Recoverable issues, deprecated usage
- `.error()` Failures that stop current operation
- `.exception()` Errors with full stack trace (use in exception handlers)

**Environment configuration:**

```bash
LOG_LEVEL=DEBUG uv run python main.py              # Set global log level
LOG_LEVEL_MODEL_LOADER=DEBUG uv run python main.py # Set module-specific level
```
