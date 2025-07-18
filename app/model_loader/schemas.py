from typing import Dict, Union

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import torch

from app.database.crud import get_device_index, get_gpu_max_memory, get_ram_max_memory
from app.services import MemoryService, device_service


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
        self.max_ram_in_gb = (memory_service.total_ram * max_ram_memory) / (1024**3)
        self.max_gpu_in_gb = (memory_service.total_gpu * max_gpu_memory) / (1024**3)

    def to_dict(self) -> Dict[Union[int, str], str]:
        if device_service.is_cuda:
            return {
                self.device_index: f'{self.max_gpu_in_gb}GB',
                'cpu': f'{self.max_ram_in_gb}GB',
            }
        else:
            return {'cpu': f'{self.max_ram_in_gb}GB'}


class DownloadCompletedResponse(BaseModel):
    """
    Response model for a completed download.
    Contains the ID of the model that was downloaded.
    """

    id: str = Field(..., description='The ID of the model that was downloaded.')
