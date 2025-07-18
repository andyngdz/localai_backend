from enum import IntEnum
from typing import List

from sqlalchemy.orm import Session

from app.database.models.model import Model
from app.database.models.config import Config


class DeviceSelection(IntEnum):
    NOT_FOUND = -2


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
