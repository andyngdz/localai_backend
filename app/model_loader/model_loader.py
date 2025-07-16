import logging
import torch
from diffusers import AutoPipelineForText2Image

from app.database.crud import get_selected_device
from app.database.service import SessionLocal
from config import BASE_CACHE_DIR

from .schemas import MaxMemoryConfig
logger = logging.getLogger(__name__)

def model_loader(id: str):
    db = SessionLocal()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_dtype = torch.float16 if device == 'cuda' else torch.float32
    logger.info(f'[Process] Loading model {id} to {device}')

    device_index = get_selected_device(db)
    max_memory = MaxMemoryConfig(device, device_index).to_dict()
    logger.info(f'[Process] Setting max memory for model {id}: {max_memory}')

    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            id,
            cache_dir=BASE_CACHE_DIR,
            low_cpu_mem_usage=True,
            max_memory=max_memory,
            torch_dtype=torch_dtype,
            use_safetensors=True,
        )
    except EnvironmentError:
        pipe = AutoPipelineForText2Image.from_pretrained(
            id,
            cache_dir=BASE_CACHE_DIR,
            low_cpu_mem_usage=True,
            max_memory=max_memory,
            torch_dtype=torch_dtype,
            use_safetensors=False,
        )

    pipe.to(device)

    if device == 'cuda':
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()

    db.close()

    return pipe
