"""Generated Images Table"""

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, TimestampMixin


class GeneratedImage(Base, TimestampMixin):
	"""Generated Images"""

	__tablename__ = 'generated_images'

	id: Mapped[int] = mapped_column(primary_key=True, index=True)
	history_id: Mapped[str] = mapped_column(ForeignKey('histories.id'))
	path: Mapped[str] = mapped_column()

	def __repr__(self):
		return (
			f"<GeneratedImage(history_id='{self.history_id}', path='{self.path}', "
			f"created_at='{self.created_at}', updated_at='{self.updated_at}')>"
		)
