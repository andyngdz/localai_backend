# Plan 006: Add LoRA Support

## Overview
Add LoRA (Low-Rank Adaptation) support to enable users to apply style/character LoRAs during image generation.

## User Workflow
1. User downloads LoRA from any source (HuggingFace, CivitAI, etc.)
2. User clicks "Add LoRA" in UI and selects file from filesystem
3. UI sends file path to backend via `POST /loras/upload`
4. Backend validates and copies file to `.cache/loras/`
5. Backend registers LoRA in database
6. User selects LoRA(s) with weights when generating images

## Architecture Decisions
- **File handling:** Copy to `.cache/loras/` (preserves original)
- **API method:** REST endpoint with JSON body `{file_path: "..."}`
- **Registration:** Explicit only (no auto-scan on startup)
- **Memory:** Unload LoRAs after each generation
- **Adapter naming:** Use database ID (`lora_{id}`) for internal tracking
- **Display name:** Original filename for UI/logs
- **Multiple LoRAs:** Support 1-5 LoRAs per generation with individual weights

## Implementation Tasks

### **1. Database Schema**
Create `loras` table via Alembic migration:
```python
class LoRA(Base, TimestampMixin):
	id: int (primary key)
	name: str (from filename, for display)
	file_path: str (path in .cache/loras/)
	file_size: int (bytes)
```

Add CRUD operations in `app/database/crud.py`

### **2. Storage Service**
Add to `app/services/storage.py`:
```python
def get_loras_dir() -> str:
	return os.path.join(CACHE_FOLDER, 'loras')

def get_lora_file_path(filename: str) -> str:
	return os.path.join(get_loras_dir(), filename)
```

### **3. Feature Module: `app/features/loras/`**
```
├── __init__.py
├── api.py           # REST endpoints
├── schemas.py       # Pydantic models
├── service.py       # Business logic
└── file_manager.py  # File copy/validation
```

**Schemas:**
```python
class LoRAUploadRequest(BaseModel):
	file_path: str

class LoRAConfigItem(BaseModel):
	lora_id: int
	weight: float = Field(default=1.0, ge=0.0, le=2.0)

class LoRAInfo(BaseModel):
	id: int
	name: str
	file_size: int
	created_at: datetime
```

**API Endpoints:**
- `POST /loras/upload` - Copy file from path
- `GET /loras` - List all LoRAs
- `DELETE /loras/{id}` - Remove LoRA
- `GET /loras/{id}` - Get details

**Service logic:**
- Validate file exists and is `.safetensors`
- Check file size limit (500MB)
- Copy to `.cache/loras/`
- Handle duplicate filenames
- Save to database

### **4. Generator Schema Update**
Modify `app/features/generators/schemas.py`:
```python
class LoRAConfigItem(BaseModel):
	lora_id: int
	weight: float = Field(default=1.0, ge=0.0, le=2.0)

class GeneratorConfig(BaseModel):
	# ... existing fields ...
	loras: list[LoRAConfigItem] = Field(default_factory=list)
```

### **5. Pipeline Manager Integration**
Add to `app/cores/model_manager/pipeline_manager.py`:
```python
def load_loras(self, lora_configs: list[dict]):
	"""Load LoRAs with weights into pipeline

	Args:
		lora_configs: List of {id, name, file_path, weight}
	"""
	adapter_names = []
	adapter_weights = []

	for config in lora_configs:
		adapter_name = f"lora_{config['id']}"

		logger.info(
			f"Loading LoRA '{config['name']}' as adapter "
			f"'{adapter_name}' (weight: {config['weight']})"
		)

		self.pipe.load_lora_weights(
			config['file_path'],
			adapter_name=adapter_name
		)
		adapter_names.append(adapter_name)
		adapter_weights.append(config['weight'])

	self.pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
	logger.info(f"Loaded {len(adapter_names)} LoRAs successfully")

def unload_loras(self):
	"""Remove all LoRAs from pipeline"""
	self.pipe.unload_lora_weights()
	logger.info("Unloaded all LoRAs")
```

### **6. Generator Service Integration**
Modify `app/features/generators/service.py`:
```python
# Before generation
if config.loras:
	lora_data = []
	for lora_config in config.loras:
		lora = await lora_crud.get_by_id(lora_config.lora_id)
		lora_data.append({
			'id': lora.id,
			'name': lora.name,
			'file_path': lora.file_path,
			'weight': lora_config.weight
		})

	model_manager.pipeline_manager.load_loras(lora_data)

try:
	# Generate images
	output = pipe(...)
finally:
	if config.loras:
		model_manager.pipeline_manager.unload_loras()
```

### **7. Testing**
- Upload valid/invalid file paths
- Upload duplicate files
- List/delete operations
- Generation with 1-5 LoRAs
- Different weight values (0.0 - 2.0)
- Error handling (missing file, corrupted LoRA)
- Memory cleanup verification
- Pre-commit hooks pass

## Files to Create/Modify

**New:**
- `app/features/loras/__init__.py`
- `app/features/loras/api.py`
- `app/features/loras/schemas.py`
- `app/features/loras/service.py`
- `app/features/loras/file_manager.py`
- `alembic/versions/{timestamp}_add_loras_table.py`
- `tests/app/features/loras/test_service.py`
- `tests/app/features/loras/test_api.py`

**Modified:**
- `app/features/generators/schemas.py` (add loras field)
- `app/features/generators/service.py` (LoRA loading/unloading)
- `app/cores/model_manager/pipeline_manager.py` (LoRA methods)
- `app/services/storage.py` (LoRA directory paths)

## Technical Details

### Diffusers API Usage
```python
# Load multiple LoRAs
pipe.load_lora_weights("path/to/lora1.safetensors", adapter_name="lora_1")
pipe.load_lora_weights("path/to/lora2.safetensors", adapter_name="lora_2")

# Set weights
pipe.set_adapters(["lora_1", "lora_2"], adapter_weights=[0.8, 0.6])

# Generate
output = pipe(prompt="...", ...)

# Cleanup
pipe.unload_lora_weights()
```

### File Format
- Format: `.safetensors` (preferred for safety and speed)
- Typical size: 10-200MB per LoRA
- Compatible sources: HuggingFace, CivitAI, Kohya, Automatic1111

### Memory Management
- Load LoRAs before each generation
- Unload immediately after completion
- Clear on model unload
- No fusion in Phase 1 (keep simple)

## Success Criteria
- ✅ Copy LoRA from user path to `.cache/loras/`
- ✅ List all registered LoRAs with metadata
- ✅ Delete LoRA (file + DB entry)
- ✅ Generate with multiple LoRAs and individual weights
- ✅ Adapter names use DB IDs for uniqueness
- ✅ Logs show friendly LoRA names for debugging
- ✅ Handle duplicate filenames gracefully
- ✅ Proper error messages for invalid files
- ✅ LoRAs unload after each generation
- ✅ All tests pass (ruff, mypy, pytest)

## Future Enhancements (Phase 2)
- LoRA fusion for faster inference
- Fine-grained weight control (per-component)
- HuggingFace download integration
- CivitAI API integration
- Smart caching (keep frequently used LoRAs loaded)
- LoRA metadata extraction (trigger words, base model compatibility)
