"""Models Table"""

from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import Mapped, mapped_column

from app.database.core import Base


class Model(Base):
    """Models"""

    __tablename__ = 'models'

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    model_id: Mapped[str] = mapped_column(unique=True)
    model_dir: Mapped[str] = mapped_column(unique=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )
