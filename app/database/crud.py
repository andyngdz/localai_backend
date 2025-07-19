from typing import List

from sqlalchemy.orm import Session

from app.database.models import Model, Config, History
from app.schemas.generators import ImageGenerationRequest
from .constant import DEFAULT_MAX_GPU_MEMORY, DEFAULT_MAX_RAM_MEMORY, DeviceSelection


def add_model(db: Session, model_id: str, model_dir: str):
    """Add a model to the database or update its local path if it already exists."""
    model = db.query(Model).filter(Model.model_id == model_id).first()

    if model:
        model.model_dir = model_dir
    else:
        model = Model(model_id=model_id, model_dir=model_dir)
        db.add(model)

    db.commit()

    return model


def downloaded_models(db: Session) -> List[Model]:
    """Get all models from the database."""
    models = db.query(Model).all()

    return models


def is_model_downloaded(db: Session, model_id: str) -> bool:
    """Check if a model is downloaded (status 'completed') in the database."""
    model = db.query(Model).filter(Model.model_id == model_id).first()

    return model is not None


def get_device_index(db: Session) -> int:
    """Get selected device index from the database."""
    config = db.query(Config).first()

    return config.device_index if config else DeviceSelection.NOT_FOUND


def add_device_index(db: Session, device_index: int):
    """Add or update selected device"""

    config = db.query(Config).first()

    if config:
        config.device_index = device_index
    else:
        config = Config(device_index=device_index)
        db.add(config)

    db.commit()


def add_max_memory(db: Session, ram: float, gpu: float):
    """Add or update configuration in the database."""
    config = db.query(Config).first()

    if config:
        config.ram = ram
        config.gpu = gpu
    else:
        config = Config(ram=ram, gpu=gpu)
        db.add(config)

    db.commit()


def get_gpu_max_memory(db: Session) -> float:
    """Get GPU max memory from the database."""

    config = db.query(Config).first()

    if config and config.gpu is not None:
        return config.gpu

    return DEFAULT_MAX_GPU_MEMORY


def get_ram_max_memory(db: Session) -> float:
    """Get RAM max memory from the database."""

    config = db.query(Config).first()

    if config and config.ram is not None:
        return config.ram

    return DEFAULT_MAX_RAM_MEMORY


def add_history(db: Session, model: str, request: ImageGenerationRequest):
    """Add a history entry to the database."""
    config = request.config

    history = History(
        prompt=config.prompt,
        model=model,
        config=config.model_dump(),
    )
    db.add(history)
    db.commit()

    return history
