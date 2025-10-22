# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Switch to .venv

```bash
source .venv/bin/activate
```

### Running the Application

```bash
python main.py
# or
uvicorn main:app --reload
```

### Testing

```bash
# Run all tests with coverage
pytest -q --cov=app --cov-report=xml:coverage.xml --cov-report=term

# Run tests with basic output
pytest

# Run specific test file
pytest tests/app/features/downloads/test_api.py

# Run with verbose output
pytest -v
```

### Code Quality

```bash
# Format code (uses single quotes, tabs for indentation)
ruff format

# Lint code
ruff check

# Lint and fix issues
ruff check --fix
```

### Database Operations

```bash
# Database migrations are handled via Alembic
# Generate migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Downgrade migration
alembic downgrade -1
```

### Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install both
pip install -r requirements.txt -r requirements-dev.txt
```

## Architecture Overview

### Application Structure

- **FastAPI Application**: Main entry point in `main.py` with lifespan management
- **Feature-based Architecture**: Code organized by features under `app/features/`
- **Core Services**: Shared business logic in `app/cores/`
- **Database Layer**: SQLAlchemy models and CRUD operations in `app/database/`
- **Service Layer**: Business logic services in `app/services/` (models, storage, etc.)
- **WebSocket Support**: Real-time communication via Socket.IO

### Key Components

**Features** (`app/features/`):

- `generators/` - AI image generation endpoints and logic
- `downloads/` - Model download management
- `hardware/` - System hardware information
- `histories/` - Generation history tracking
- `models/` - Model management APIs
- `styles/` - Predefined image styles
- `users/` - User management
- `resizes/` - Image resizing functionality

**Core Services** (`app/cores/`):

- `model_manager/` - AI model loading and management
- `model_loader/` - Model loading utilities with device optimization
- `samplers/` - Sampling algorithms and schedulers
- `constants/` - Shared constants and configurations

**Database Models** (`app/database/models/`):

- `Model` - AI model metadata (model_id, model_dir)
- `GeneratedImage` - Generated image records
- `History` - Generation history entries
- `Config` - Application configuration

### Service Layer

- `app/services/` contains utility services:
  - `device.py` - GPU/CPU device management
  - `image.py` - Image processing utilities
  - `storage.py` - File storage management
  - `logger.py` - Logging configuration
  - `memory.py` - Memory usage monitoring

### Configuration

- Database: SQLite (`localai_backend.db`)
- Static files served from `static/` directory
- Generated images stored in `static/generated_images/`
- Cache directory: `.cache/`
- CORS enabled for all origins

### Development Notes

- Uses tab indentation and single quotes (configured in `ruff.toml`)
- Async/await patterns throughout FastAPI endpoints
- SQLAlchemy 2.0 with declarative base
- Comprehensive test coverage with pytest and pytest-asyncio
- CI/CD pipeline includes code quality checks and SonarCloud analysis
