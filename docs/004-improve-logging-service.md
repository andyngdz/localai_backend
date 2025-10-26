# 004: Improve Logging Service

## Overview

Enhanced the logging service to provide better readability, clear messages, separate loggers per module, and consistent category grouping.

## Goals

✅ Separate loggers per feature/module for better filtering
✅ Consistent category prefixes (`[ModelLoad]`, `[Download]`, etc.)
✅ Colored console output for better readability
✅ Enhanced formatting with clear structure

## Implementation Summary

### 1. Core Logger Service (`app/services/logger.py`)

**Changes Made:**
- Created `CategoryAdapter` class for automatic category prefix injection
- Implemented `LoggerService` class with:
  - `init()` method for one-time root logger configuration
  - `get_logger(name, category=None)` helper method
  - Environment variable support for log level configuration
- Removed module names from output (redundant with categories)
- Added colored formatter using `colorlog`:
  - DEBUG = cyan
  - INFO = green
  - WARNING = yellow
  - ERROR = red
  - CRITICAL = red,bold
- Entire log line colored based on severity level
- Suppressed verbose third-party loggers (socketio, engineio)
- Applied formatter to uvicorn loggers for consistency

**Key Features:**
- `LOG_LEVEL` - Global log level (DEBUG, INFO, WARNING, ERROR)
- `LOG_LEVEL_LAI_<MODULE>` - Per-module log level override
  - Example: `LOG_LEVEL_LAI_MODEL_LOADER=DEBUG`
  - Module name conversion: `app.cores.model_loader` → `LAI_MODEL_LOADER`

### 2. Log Format

**Final Format:**
```
[LEVEL] YYYY-MM-DD HH:MM:SS  [Category] message
```

**Example Output:**
```
[INFO] 2025-10-26 10:44:16  [ModelLoad] Loading model RunDiffusion/Juggernaut-XL-v9 to cuda
[INFO] 2025-10-26 10:44:16  [ModelLoad] RunDiffusion/Juggernaut-XL-v9 step=1/9 phase=initialization
[INFO] 2025-10-26 10:44:16  [Download] Starting download for model flux-dev
[INFO] 2025-10-26 10:44:16  [Generate] Generating 4 images with seed 12345
[INFO] 2025-10-26 10:44:16  [API] GET /models/downloaded - 200 OK
```

**Removed:**
- Module names (e.g., `app.cores.model_loader.model_loader`) - too verbose
- Trailing spaces after log level
- Duplicate socketio logging messages
- Plain uvicorn access logs (now colored)

### 3. Standard Categories

**Implemented Categories:**

| Category | Files | Purpose |
|----------|-------|---------|
| `[ModelLoad]` | 6 files | Model loading, state management, resource cleanup |
| `[Download]` | 2 files | Model download operations |
| `[Generate]` | 6 files | Image generation, memory management, progress |
| `[API]` | 5 files | API endpoint requests/responses |
| (none) | 7 files | Utility services without specific domain |

### 4. Files Updated (26 total)

**ModelLoad Category (6 files):**
1. `app/cores/model_loader/model_loader.py`
2. `app/cores/model_manager/loader_service.py`
3. `app/cores/model_manager/model_manager.py`
4. `app/cores/model_manager/pipeline_manager.py`
5. `app/cores/model_manager/resource_manager.py`
6. `app/cores/model_manager/state_manager.py`

**Download Category (2 files):**
7. `app/features/downloads/services.py`
8. `app/features/downloads/api.py`

**Generate Category (6 files):**
9. `app/features/generators/service.py`
10. `app/features/generators/api.py`
11. `app/cores/generation/image_processor.py`
12. `app/cores/generation/progress_callback.py`
13. `app/cores/generation/memory_manager.py`
14. `app/cores/generation/seed_manager.py`

**API Category (5 files):**
15. `app/features/models/api.py`
16. `app/features/hardware/api.py`
17. `app/features/histories/api.py`
18. `app/features/img2img/api.py`
19. `app/features/resizes/api.py`

**No Category (7 files):**
20. `app/features/img2img/service.py`
21. `app/features/models/recommendations.py`
22. `app/cores/gpu_utils.py`
23. `app/cores/pipeline_converter/pipeline_converter.py`
24. `app/services/device.py`
25. `app/services/styles.py`
26. `app/database/service.py`

**Pattern Applied:**
```python
# Old
import logging
logger = logging.getLogger(__name__)

# New
from app.services import logger_service
logger = logger_service.get_logger(__name__, category='CategoryName')
```

### 5. Additional Changes

**Dependencies:**
- Added `colorlog==6.9.0` to `pyproject.toml`

**Main Application:**
- Updated `main.py:38` to call `logger_service.init()`

**Import Order:**
- Fixed circular import in `app/services/__init__.py` by importing `logger_service` first

**Socket Service:**
- Disabled SocketIO's built-in logger (`logger=False` in `app/socket/service.py:27`)

**Tests:**
- Updated `tests/app/cores/model_loader/test_model_load_progress.py` to not check for hardcoded `[ModelLoad]` prefix (now added by CategoryAdapter)

**Documentation:**
- Updated `CLAUDE.md` with comprehensive logging guidelines

### 6. Breaking Changes

**None** - Fully backward compatible:
- Existing `logging.getLogger(__name__)` calls still work
- Standard Python logging functions work as expected
- All 478 tests pass

## Configuration Examples

**Global log level:**
```bash
LOG_LEVEL=DEBUG python main.py
```

**Module-specific log level:**
```bash
LOG_LEVEL_LAI_MODEL_LOADER=DEBUG python main.py
```

**Multiple modules:**
```bash
LOG_LEVEL=INFO LOG_LEVEL_LAI_MODEL_LOADER=DEBUG LOG_LEVEL_LAI_DOWNLOADS=DEBUG python main.py
```

## Usage Examples

**Basic logging (no category):**
```python
from app.services import logger_service
logger = logger_service.get_logger(__name__)
logger.info('Operation completed')
```

**With category prefix:**
```python
from app.services import logger_service
logger = logger_service.get_logger(__name__, category='ModelLoad')
logger.info('Loading model...')  # Output: [INFO] ... [ModelLoad] Loading model...
```

## Technical Details

### CategoryAdapter Implementation

```python
class CategoryAdapter(logging.LoggerAdapter):
    def __init__(self, logger: logging.Logger, category: str):
        super().__init__(logger, {})
        self.category = category

    def process(self, msg: str, kwargs):
        return f'[{self.category}] {msg}', kwargs
```

### Formatter Configuration

```python
formatter = colorlog.ColoredFormatter(
    '%(log_color)s[%(levelname)s] %(asctime)s  %(message)s%(reset)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bold',
    },
)
```

## Success Criteria

- ✅ Colored console output with clear visual hierarchy
- ✅ Consistent category prefixes across 26 modules
- ✅ Per-module log level control via environment variables
- ✅ Backward compatible with existing code
- ✅ All 478 tests pass
- ✅ No circular imports
- ✅ Documentation updated (CLAUDE.md)
- ✅ Clean, compact output without redundant information
- ✅ Unified format for all loggers (including uvicorn)

## Before vs After

**Before:**
```
INFO:     127.0.0.1:35344 - "WebSocket /socket.io/?EIO=4&transport=websocket" [accepted]
[INFO    ] 2025-10-26 10:26:16  app.cores.model_loader.model_loader  Loading model...
[INFO    ] 2025-10-26 10:26:16  app.cores.model_loader.model_loader  [ModelLoad] step=1/9...
emitting event "SocketEvents.MODEL_LOAD_PROGRESS" to all [/]
[INFO    ] 2025-10-26 10:26:16  socketio.server                 emitting event "SocketEvents.MODEL_LOAD_PROGRESS" to all [/]
```

**After:**
```
[INFO] 2025-10-26 10:26:16  127.0.0.1:35344 - "WebSocket /socket.io/?EIO=4&transport=websocket" [accepted]
[INFO] 2025-10-26 10:26:16  [ModelLoad] Loading model RunDiffusion/Juggernaut-XL-v9 to cuda
[INFO] 2025-10-26 10:26:16  [ModelLoad] RunDiffusion/Juggernaut-XL-v9 step=1/9 phase=initialization
```

Clean, colored, and consistent!
