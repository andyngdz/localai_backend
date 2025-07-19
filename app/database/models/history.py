"""Histories Table"""

from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class Model(Base):
    """Histories"""

    __tablename__ = 'histories'

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    prompt: Mapped[str] = mapped_column()
    model: Mapped[str] = mapped_column()
    config: Mapped[str] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    def __repr__(self):
        return f"<History(prompt='{self.prompt}', model='{self.model}', config='{self.config}', created_at='{self.created_at}')>"
