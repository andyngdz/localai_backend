# python_backend/app/database/crud.py

from enum import IntEnum
from typing import List

from sqlalchemy.orm import Session

from app.database.models.model import Model
from app.database.models.user import User


class DeviceSelection(IntEnum):
    NOT_FOUND = -2


def get_selected_device(db: Session) -> int:
    """Get selected device index from the database."""
    user = db.query(User).first()

    return user.selected_device if user else DeviceSelection.NOT_FOUND


def add_model(db: Session, model_id: str, model_dir: str):
    """Add a model to the database or update its local path if it already exists."""
    model = db.query(Model).filter(Model.model_id == model_id).first()

    if model:
        model.model_dir = model_dir  # Update path if already exists
    else:
        model = Model(model_id=model_id, model_dir=model_dir)
        db.add(model)

    db.commit()
    db.refresh(model)

    return model


def get_all_downloaded_models(db: Session) -> List[Model]:
    """Get all models from the database."""
    models = db.query(Model).all()

    return models


def check_if_model_downloaded(db: Session, model_id: str) -> bool:
    """Check if a model is downloaded (status 'completed') in the database."""
    model = db.query(Model).filter(Model.model_id == model_id).first()

    return model is not None


def create_or_update_selected_device(db: Session, device_index: int):
    """Create or update selected device"""

    user = db.query(User).first()

    if user:
        user.selected_device = device_index
    else:
        user = User(selected_device=device_index)
        db.add(user)

    db.commit()
