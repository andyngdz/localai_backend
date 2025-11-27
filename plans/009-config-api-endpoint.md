# Plan: Create /config API Endpoint

## Overview
Create a new `/config` endpoint that returns configuration data to the frontend, starting with `UpscalerType` options with metadata (value, name, description, suggested_denoise_strength).

## Files Created

### 1. `app/schemas/config.py` - Response schemas
- `UpscalerItem` - Upscaler option with value, name, description, suggested_denoise_strength
- `ConfigResponse` - Response containing list of upscalers

### 2. `app/features/config/` - Feature module
- `service.py` - ConfigService with UPSCALER_METADATA dictionary
- `api.py` - GET /config endpoint
- `__init__.py` - Module export

### 3. `tests/app/features/config/` - Tests
- `test_service.py` - Service unit tests
- `test_api.py` - API endpoint tests

## Files Modified

### `main.py`
- Added import for config router
- Registered config router with `app.include_router(config)`

## API Response

```json
{
  "upscalers": [
    {"value": "Lanczos", "name": "Lanczos (High Quality)", "description": "High-quality resampling, best for photos", "suggested_denoise_strength": 0.4},
    {"value": "Bicubic", "name": "Bicubic (Smooth)", "description": "Smooth interpolation, good balance", "suggested_denoise_strength": 0.4},
    {"value": "Bilinear", "name": "Bilinear (Fast)", "description": "Fast interpolation, moderate quality", "suggested_denoise_strength": 0.35},
    {"value": "Nearest", "name": "Nearest (Sharp Edges)", "description": "No interpolation, preserves sharp edges", "suggested_denoise_strength": 0.3}
  ]
}
```

## Denoise Strength Rationale
Values optimized for composition preservation based on upscaler characteristics:
- **Lanczos (0.4)** - Sharp results can handle moderate refinement
- **Bicubic (0.4)** - Good all-around balance
- **Bilinear (0.35)** - Softer output, needs less modification
- **Nearest (0.3)** - Preserves hard pixel edges, minimal change needed

## Extensibility
The `ConfigResponse` schema can be extended with additional fields as more config items are needed.
