"""Users Table"""

from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class User(Base):
    """Users"""

    __tablename__ = 'users'

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    selected_device: Mapped[int] = mapped_column()

    def __repr__(self):
        return f'<User(id={self.id}, selected_device={self.selected_device})>'
