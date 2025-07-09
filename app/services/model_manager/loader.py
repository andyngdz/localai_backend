import logging
from multiprocessing import Queue
from typing import Optional

import torch
from diffusers import AutoPipelineForText2Image

from config import BASE_CACHE_DIR

logger = logging.getLogger(__name__)


def load_model_process(id: str, device: str, done_queue: Optional[Queue] = None):
    logger.info(f'[Process] Loading model {id} to {device}')

    max_memory = {0: '8GB', 'cpu': '8GB'} if device == 'cuda' else {'cpu': '8GB'}

    logger.info(f'[Process] Setting max memory for model {id}: {max_memory}')

    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            id,
            cache_dir=BASE_CACHE_DIR,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            max_memory=max_memory,
        )
    except EnvironmentError:
        pipe = AutoPipelineForText2Image.from_pretrained(
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
