# python_backend/app/database/__init__.py


import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.base import Base

# Define the database URL
DATABASE_URL = 'sqlite:///localai_backend.db'

# Create the SQLAlchemy engine
# echo=True is useful for debugging SQL queries, set to False in production
engine = create_engine(DATABASE_URL, echo=False)

# Create a SessionLocal class, which is a factory for new Session objects
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


logger = logging.getLogger(__name__)


# Function to initialize the database schema
def init_db():
    """
    Initializes the database schema by creating tables.
    This should be called only once, typically on application startup.
    """
    # Import all models here so that Base.metadata knows about them
    # This is crucial for Base.metadata.create_all() to find all your defined tables
    from app.database.models import (
        Model,
        User,
    )

    # Create all tables defined by Base.metadata
    Base.metadata.create_all(bind=engine)
    logger.info('Database initialized successfully.')


def get_db():
    """
    Dependency to get a database session for a request.
    It ensures the session is closed after the request is finished.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
