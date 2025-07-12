import logging
from multiprocessing import Queue
from typing import Optional

import torch
from diffusers import AutoPipelineForText2Image
from sqlalchemy.orm import Session

from app.database.crud import get_selected_device
from config import BASE_CACHE_DIR

from .schemas import MaxMemoryConfig

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    A class to handle the loading of models in a separate process.
    This is useful for managing resources and avoiding blocking the main application.
    """

    def __init__(self):
        self.pipe = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.torch_dtype = torch.float16 if self.device == 'cuda' else torch.float32

    def process(
        self,
        id: str,
        db: Session,
        done_queue: Optional[Queue] = None,
    ):
        logger.info(f'[Process] Loading model {id} to {self.device}')

        device_index = get_selected_device(db)
        max_memory = MaxMemoryConfig(self.device, device_index).to_dict()
        logger.info(f'[Process] Setting max memory for model {id}: {max_memory}')

        try:
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                id,
                cache_dir=BASE_CACHE_DIR,
                low_cpu_mem_usage=True,
                max_memory=max_memory,
                torch_dtype=self.torch_dtype,
                use_safetensors=True,
            )
        except EnvironmentError:
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                id,
                cache_dir=BASE_CACHE_DIR,
                low_cpu_mem_usage=True,
                max_memory=max_memory,
                torch_dtype=self.torch_dtype,
                use_safetensors=False,
            )

        self.pipe.to(self.device)

        if self.device == 'cuda':
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_attention_slicing()

        if done_queue is not None:
            done_queue.put(id)
            logger.info(f'[Process] Model {id} added to done queue.')

        return self.pipe


model_loader = ModelLoader()
