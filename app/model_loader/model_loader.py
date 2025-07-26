import logging

from diffusers import AutoPipelineForText2Image

from app.database.crud import add_model
from app.database.service import SessionLocal
from app.services import device_service, storage_service
from app.socket import SocketEvents, socket_service
from config import CACHE_DIR

from .constants import DEVICE_MAP
from .max_memory import MaxMemoryConfig
from .schemas import DownloadCompletedResponse

logger = logging.getLogger(__name__)


def model_loader(id: str):
	db = SessionLocal()

	logger.info(f'Loading model {id} to {device_service.device}')

	max_memory = MaxMemoryConfig(db).to_dict()
	logger.info(f'Max memory configuration: {max_memory}')

	logger.info(f'Device map for model {id}: {DEVICE_MAP}')

	try:
		pipe = AutoPipelineForText2Image.from_pretrained(
			id,
			cache_dir=CACHE_DIR,
			low_cpu_mem_usage=True,
			max_memory=max_memory,
			torch_dtype=device_service.torch_dtype,
			use_safetensors=True,
			device_map=DEVICE_MAP,
		)
	except EnvironmentError:
		pipe = AutoPipelineForText2Image.from_pretrained(
			id,
			cache_dir=CACHE_DIR,
			low_cpu_mem_usage=True,
			max_memory=max_memory,
			torch_dtype=device_service.torch_dtype,
			use_safetensors=False,
			device_map=DEVICE_MAP,
		)
	except Exception as error:
		logger.error(f'Error loading model {id}: {error}')
		raise

	if device_service.is_cuda:
		pipe.enable_model_cpu_offload()
		pipe.enable_attention_slicing()

	model_dir = storage_service.get_model_dir(id)

	add_model(db, id, model_dir)

	db.close()

	socket_service.emit_sync(
		SocketEvents.MODEL_LOAD_COMPLETED,
		DownloadCompletedResponse(id=id).model_dump(),
	)

	return pipe
