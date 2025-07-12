import logging
from multiprocessing import Queue
from typing import Optional, Union

import torch
from diffusers import (
    AutoPipelineForText2Image,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)
from sqlalchemy.orm import Session

from app.database.crud import get_selected_device
from config import BASE_CACHE_DIR

from .schemas import MaxMemoryConfig

logger = logging.getLogger(__name__)


SupportedPipelines = Union[
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
]


def model_loader_process(
    id: str,
    device: str,
    db: Session,
    done_queue: Optional[Queue] = None,
):
    logger.info(f'[Process] Loading model {id} to {device}')

    device_index = get_selected_device(db)
    max_memory = MaxMemoryConfig(device, device_index).to_dict()
    logger.info(f'[Process] Setting max memory for model {id}: {max_memory}')

    try:
        pipe: SupportedPipelines = AutoPipelineForText2Image.from_pretrained(
            id,
            cache_dir=BASE_CACHE_DIR,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            max_memory=max_memory,
        )
    except EnvironmentError:
        pipe: SupportedPipelines = AutoPipelineForText2Image.from_pretrained(
            id,
            cache_dir=BASE_CACHE_DIR,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            use_safetensors=False,
            low_cpu_mem_usage=True,
            max_memory=max_memory,
        )

    pipe.to(device)

    if device == 'cuda':
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()

    if done_queue is not None:
        done_queue.put(id)
        logger.info(f'[Process] Model {id} added to done queue.')

    return pipe
