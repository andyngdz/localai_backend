"""Database to store data for LocalAI Backend"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.services.database_orms.user import User

Base = declarative_base()

DATABASE_URL = "sqlite:///localai_backend.db"

engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


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


if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
