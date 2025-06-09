"""Database to store data for LocalAI Backend"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.core import Base
from app.database.user import User

DATABASE_URL = "sqlite:///localai_backend.db"

engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


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
