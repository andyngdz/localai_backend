"""Users Table"""

from sqlalchemy.orm import Mapped, mapped_column

from app.database.core import Base


class User(Base):
    """Users"""

    __tablename__ = 'users'

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    selected_device: Mapped[int] = mapped_column()
