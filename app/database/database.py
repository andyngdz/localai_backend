"""Database to store data for LocalAI Backend"""

from enum import IntEnum

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.core import Base
from app.database.user import User

DATABASE_URL = 'sqlite:///localai_backend.db'

engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


class DeviceSelection(IntEnum):
    NOT_FOUND = -2


def create_or_update_selected_device(device_index: int):
    """Create or update selected device"""
    db = SessionLocal()

    try:
        user = db.query(User).first()

        if user:
            user.selected_device = device_index
        else:
            user = User(selected_device=device_index)
            db.add(user)

        db.commit()
    finally:
        db.close()


def get_selected_device() -> int:
    """Get selected device index"""
    db = SessionLocal()

    try:
        user = db.query(User).first()
        return user.selected_device if user else DeviceSelection.NOT_FOUND
    finally:
        db.close()
