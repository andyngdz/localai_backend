"""Histories Table"""

from typing import Any

from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, TimestampMixin


class History(Base, TimestampMixin):
	"""Histories"""

	__tablename__ = 'histories'

	id: Mapped[int] = mapped_column(primary_key=True, index=True)
	prompt: Mapped[str] = mapped_column()
	model: Mapped[str] = mapped_column()
	config: Mapped[dict[str, Any]] = mapped_column()

	def __repr__(self):
		return f"<History(prompt='{self.prompt}', model='{self.model}', config='{self.config}', created_at='{self.created_at}', updated_at='{self.updated_at}')>"
