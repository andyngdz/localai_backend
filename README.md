# LocalAI Backend

A FastAPI-based backend server for local AI image generation operations, supporting Stable Diffusion models, style management, and real-time generation updates via WebSocket.

## Features

- **AI Image Generation**: Generate images using Stable Diffusion models with customizable parameters
- **Model Management**: Download, load, and manage multiple AI models locally
- **Style System**: Apply predefined styles with prompt templates and negative prompts
- **Hardware Monitoring**: Real-time GPU/CPU usage and memory tracking
- **Generation History**: Track and retrieve past image generations
- **Image Resizing**: Batch resize generated images
- **Real-time Updates**: WebSocket support for live generation progress
- **User Management**: Basic user authentication and profiles
- **SQLite Database**: Persistent storage with Alembic migrations

## Tech Stack

- **FastAPI** - Modern async web framework
- **PyTorch** - Deep learning framework
- **Diffusers** - Hugging Face diffusion models library
- **Transformers** - Model tokenizers and utilities
- **SQLAlchemy** - Database ORM with Alembic migrations
- **Socket.IO** - Real-time bidirectional communication
- **Pydantic** - Data validation and settings management
- **Uvicorn** - ASGI server

## Requirements

- Python 3.11+
- CUDA-compatible GPU (optional, for faster generation)
- 8GB+ RAM recommended
- 50GB+ disk space for models (some models can exceed 50GB)

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd localai_backend
```

### 2. Install uv

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

More installation methods: [uv documentation](https://docs.astral.sh/uv/getting-started/installation/)

### 3. Install dependencies

```bash
uv sync
```

### 4. Initialize database

```bash
uv run alembic upgrade head
```

## Running the Server

### Development mode with auto-reload

```bash
uv run uvicorn main:app --reload
```

### Production mode

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`.

## Project Structure

```
localai_backend/
├── main.py                     # FastAPI application entry point
├── config.py                   # Configuration settings
├── app/
│   ├── features/               # Feature modules (routers)
│   │   ├── downloads/          # Model downloading
│   │   ├── generators/         # Image generation
│   │   ├── hardware/           # System monitoring
│   │   ├── histories/          # Generation history
│   │   ├── models/             # Model management
│   │   ├── resizes/            # Image resizing
│   │   ├── styles/             # Style management
│   │   └── users/              # User management
│   ├── cores/                  # Core services
│   │   ├── model_manager.py    # Model loading/unloading
│   │   └── samplers.py         # Sampling algorithms
│   ├── database/               # Database models and service
│   ├── services/               # Utility services
│   ├── socket/                 # Socket.IO integration
│   ├── schemas/                # Pydantic schemas
│   └── styles/                 # Style definitions
├── alembic/                    # Database migrations
├── static/                     # Static files and generated images
├── tests/                      # Test suite
└── requirements.txt            # Python dependencies
```

## API Endpoints

### Core Endpoints

- `GET /` - Health check
- `GET /docs` - OpenAPI documentation (Swagger UI)

### Features

- `/users` - User management
- `/models` - AI model operations (list, load, unload)
- `/downloads` - Download models from Hugging Face
- `/hardware` - System resource monitoring
- `/generators` - Image generation with parameters
- `/styles` - Style templates management
- `/histories` - Generation history tracking
- `/resizes` - Image resizing operations

### WebSocket

- `/socket.io` - Real-time generation progress and updates

## Development

### Running tests

```bash
uv run pytest -q --cov=app --cov-report=xml:coverage.xml
```

### Linting and formatting

```bash
# Check code style
uv run ruff check

# Auto-fix issues
uv run ruff check --fix

# Format code
uv run ruff format
```

### Database migrations

```bash
# Create a new migration
uv run alembic revision --autogenerate -m "description of changes"

# Apply migrations
uv run alembic upgrade head

# Rollback migration
uv run alembic downgrade -1
```

## Configuration

Configuration is managed through `config.py` with the following key settings:

- `CACHE_FOLDER`: Model cache directory (default: `./.cache`)
- `GENERATED_IMAGES_FOLDER`: Output directory for generated images
- `STATIC_FOLDER`: Static files directory

Override settings using environment variables as needed.

## Contributing

1. Follow the coding style defined in `ruff.toml` (tabs, single quotes, 120 char line length)
2. Use conventional commit messages (`feat:`, `fix:`, `refactor:`, etc.)
3. Write tests for new features
4. Ensure all tests pass and code is linted before submitting PRs

See `AGENTS.md` for detailed development guidelines.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Powered by [Hugging Face Diffusers](https://huggingface.co/docs/diffusers)
- Real-time updates via [python-socketio](https://python-socketio.readthedocs.io/)
