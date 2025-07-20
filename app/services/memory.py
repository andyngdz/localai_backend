import psutil
import torch
from sqlalchemy.orm import Session

from app.database.crud import get_device_index


class MemoryService:
	def __init__(self, db: Session):
		device_index = get_device_index(db)
		properties = torch.cuda.get_device_properties(device_index)
		total_ram = psutil.virtual_memory().total

		self.total_ram = total_ram
		self.total_gpu = properties.total_memory
