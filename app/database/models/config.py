"""Config Table"""

from sqlalchemy import Float, INT
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class Config(Base):
    """Config"""

    __tablename__ = 'config'

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    device_index: Mapped[int] = mapped_column()
    ram: Mapped[float] = mapped_column(default=0.5, nullable=True)
    gpu: Mapped[float] = mapped_column(default=0.5, nullable=True)

    def __repr__(self):
        return f'<Config(id={self.id}, device_index={self.device_index}, ram={self.ram}, gpu={self.gpu})>'
