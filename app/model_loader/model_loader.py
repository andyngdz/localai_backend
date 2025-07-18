import logging
from diffusers import AutoPipelineForText2Image

from app.database.crud import add_model
from app.database.service import SessionLocal
from app.services import device_service, get_model_dir
from app.socket import SocketEvents, socket_service

from config import BASE_CACHE_DIR

from .schemas import MaxMemoryConfig, DownloadCompletedResponse

logger = logging.getLogger(__name__)

DEVICE_MAP = 'balanced'


def model_loader(id: str):
    db = SessionLocal()

    logger.info(f'[Process] Loading model {id} to {device_service.device}')

    max_memory = MaxMemoryConfig(db).to_dict()
    logger.info(f'[Process] Max memory configuration: {max_memory}')

    logger.info(f'[Process] Device map for model {id}: {DEVICE_MAP}')

    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            id,
            cache_dir=BASE_CACHE_DIR,
            low_cpu_mem_usage=True,
            max_memory=max_memory,
            torch_dtype=device_service.torch_dtype,
            use_safetensors=True,
            device_map=DEVICE_MAP,
        )
    except EnvironmentError:
        pipe = AutoPipelineForText2Image.from_pretrained(
            id,
            cache_dir=BASE_CACHE_DIR,
            low_cpu_mem_usage=True,
            max_memory=max_memory,
            torch_dtype=device_service.torch_dtype,
            use_safetensors=False,
            device_map=DEVICE_MAP,
        )

    if device_service.is_cuda:
        pipe.enable_attention_slicing()

    model_dir = get_model_dir(id)

    add_model(db, id, model_dir)

    db.close()

    socket_service.emit_sync(
        SocketEvents.DOWNLOAD_COMPLETED,
        DownloadCompletedResponse(id=id).model_dump(),
    )

    return pipe
