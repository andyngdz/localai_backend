import logging

from diffusers import AutoPipelineForText2Image
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor

from app.database.crud import add_model
from app.database.service import SessionLocal
from app.model_loader.constants import CLIP_IMAGE_PROCESSOR_MODEL, SAFETY_CHECKER_MODEL
from app.services import device_service, storage_service
from app.socket import SocketEvents, socket_service
from config import CACHE_FOLDER

from .max_memory import MaxMemoryConfig
from .schemas import ModelLoadCompletedResponse

logger = logging.getLogger(__name__)


def model_loader(id: str):
	db = SessionLocal()

	logger.info(f'Loading model {id} to {device_service.device}')

	max_memory = MaxMemoryConfig(db).to_dict()
	logger.info(f'Max memory configuration: {max_memory}')

	feature_extractor = CLIPImageProcessor.from_pretrained(CLIP_IMAGE_PROCESSOR_MODEL)
	safety_checker_instance = StableDiffusionSafetyChecker.from_pretrained(SAFETY_CHECKER_MODEL)

	try:
		pipe = AutoPipelineForText2Image.from_pretrained(
			id,
			cache_dir=CACHE_FOLDER,
			low_cpu_mem_usage=True,
			max_memory=max_memory,
			torch_dtype=device_service.torch_dtype,
			use_safetensors=True,
			safety_checker=safety_checker_instance,
			feature_extractor=feature_extractor,
		)
	except EnvironmentError:
		pipe = AutoPipelineForText2Image.from_pretrained(
			id,
			cache_dir=CACHE_FOLDER,
			low_cpu_mem_usage=True,
			max_memory=max_memory,
			torch_dtype=device_service.torch_dtype,
			use_safetensors=False,
			safety_checker=safety_checker_instance,
			feature_extractor=feature_extractor,
		)
	except Exception as error:
		logger.error(f'Error loading model {id}: {error}')
		raise

	# Apply device-specific optimizations
	if device_service.is_cuda:
		pipe.enable_model_cpu_offload()
		pipe.enable_attention_slicing()
	elif device_service.is_mps:
		# For MPS, we can enable attention slicing but not model CPU offload
		# as it's not supported/needed on Apple Silicon
		pipe.enable_attention_slicing()
		logger.info('Applied MPS optimizations: attention slicing enabled')

	# Move pipeline to the appropriate device
	pipe = pipe.to(device_service.device)

	model_dir = storage_service.get_model_dir(id)

	add_model(db, id, model_dir)

	db.close()

	socket_service.emit_sync(
		SocketEvents.MODEL_LOAD_COMPLETED,
		ModelLoadCompletedResponse(id=id).model_dump(),
	)

	return pipe
