import logging

import torch
from diffusers import AutoPipelineForText2Image

from config import BASE_CACHE_DIR

logger = logging.getLogger(__name__)


def load_model_process(id: str, device: str):
    print(f'[Process] Downloading model {id} to {device}...')

    pipe = AutoPipelineForText2Image.from_pretrained(
        id,
        cache_dir=BASE_CACHE_DIR,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        use_safetensors=True,
    )

    pipe.to(device)

    if device == 'cuda':
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()

    logger.info(f'[Process] Model {id} download complete.')

    return pipe
