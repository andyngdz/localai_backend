import logging
import torch
from diffusers import AutoPipelineForText2Image

from app.database.crud import add_model, get_device_index
from app.database.service import SessionLocal
from app.services.storage import get_model_dir
from app.socket import SocketEvents, socket_service

from config import BASE_CACHE_DIR

from .schemas import MaxMemoryConfig, DownloadCompletedResponse

logger = logging.getLogger(__name__)

DEVICE_MAP = 'balanced'


def model_loader(id: str):
    db = SessionLocal()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_dtype = torch.float16 if device == 'cuda' else torch.float32
    logger.info(f'[Process] Loading model {id} to {device}')

    device_index = get_device_index(db)
    max_memory = MaxMemoryConfig(device, device_index).to_dict()
    logger.info(f'[Process] Setting max memory for model {id}: {max_memory}')

    logger.info(f'[Process] Devide map for model {id}: {DEVICE_MAP}')

    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            id,
            cache_dir=BASE_CACHE_DIR,
            low_cpu_mem_usage=True,
            max_memory=max_memory,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            device_map=DEVICE_MAP,
        )
    except EnvironmentError:
        pipe = AutoPipelineForText2Image.from_pretrained(
            id,
            cache_dir=BASE_CACHE_DIR,
            low_cpu_mem_usage=True,
            max_memory=max_memory,
            torch_dtype=torch_dtype,
            use_safetensors=False,
            device_map=DEVICE_MAP,
        )

    if device == 'cuda':
        pipe.enable_attention_slicing()

    model_dir = get_model_dir(id)

    add_model(db, id, model_dir)

    db.close()

    socket_service.emit_sync(
        SocketEvents.DOWNLOAD_COMPLETED,
        DownloadCompletedResponse(id=id).model_dump(),
    )

    return pipe
