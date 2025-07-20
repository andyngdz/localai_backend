from typing import Dict, Union

from sqlalchemy.orm import Session

from app.database.crud import get_device_index, get_gpu_max_memory, get_ram_max_memory
from app.services import MemoryService, device_service

from .constants import BYTES_TO_GB


class MaxMemoryConfig:
	def __init__(self, db: Session):
		"""
		device: 'cuda' or 'cpu'
		device_index: index of the CUDA device (default 0)
		max_ram: string representation of max RAM memory
		max_gpu: string representation of max GPU memory
		"""
		memory_service = MemoryService(db)
		max_gpu_memory = get_gpu_max_memory(db)
		max_ram_memory = get_ram_max_memory(db)

		self.device_index = get_device_index(db)
		self.max_ram_in_gb = (memory_service.total_ram * max_ram_memory) / BYTES_TO_GB
		self.max_gpu_in_gb = (memory_service.total_gpu * max_gpu_memory) / BYTES_TO_GB

	def to_dict(self) -> Dict[Union[int, str], str]:
		if device_service.is_cuda:
			return {
				self.device_index: f'{self.max_gpu_in_gb}GB',
				'cpu': f'{self.max_ram_in_gb}GB',
			}
		else:
			return {'cpu': f'{self.max_ram_in_gb}GB'}
