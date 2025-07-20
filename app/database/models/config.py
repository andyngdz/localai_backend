"""Config Table"""

from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base
from app.database.constant import DEFAULT_MAX_GPU_MEMORY, DEFAULT_MAX_RAM_MEMORY


class Config(Base):
	"""Config"""

	__tablename__ = 'config'

	id: Mapped[int] = mapped_column(primary_key=True, index=True)
	device_index: Mapped[int] = mapped_column()
	ram: Mapped[float] = mapped_column(default=DEFAULT_MAX_RAM_MEMORY)
	gpu: Mapped[float] = mapped_column(default=DEFAULT_MAX_GPU_MEMORY)

	def __repr__(self):
		return f'<Config(id={self.id}, device_index={self.device_index}, ram={self.ram}, gpu={self.gpu})>'
