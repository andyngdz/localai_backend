import logging

from diffusers import AutoPipelineForText2Image
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor

from app.cores.constants.model_loader import CLIP_IMAGE_PROCESSOR_MODEL, SAFETY_CHECKER_MODEL
from app.cores.max_memory import MaxMemoryConfig
from app.database.service import SessionLocal
from app.services import device_service
from app.socket import socket_service
from config import CACHE_FOLDER

from .schemas import ModelLoadCompletedResponse, ModelLoadFailed

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
		try:
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
			logger.error(f'Failed to load model {id}: {error}')
			socket_service.model_load_failed(ModelLoadFailed(id=id, error=str(error)))
			raise error
	except Exception as error:
		logger.error(f'Error loading model {id}: {error}')
		socket_service.model_load_failed(ModelLoadFailed(id=id, error=str(error)))
		raise

	# Apply device-specific optimizations
	if device_service.is_cuda:
		pipe.enable_model_cpu_offload()
		pipe.enable_attention_slicing()
		logger.info('Applied CUDA optimizations: CPU offloading and attention slicing enabled')
		# Note: When using CPU offloading, do NOT manually move pipeline to device
		# The offloading system will handle device placement automatically
	elif device_service.is_mps:
		# For MPS, we can enable attention slicing but not model CPU offload
		# as it's not supported/needed on Apple Silicon
		pipe.enable_attention_slicing()
		pipe = pipe.to(device_service.device)
		logger.info('Applied MPS optimizations: attention slicing enabled, moved to MPS device')
	else:
		# For CPU-only systems, keep pipeline on CPU
		pipe = pipe.to(device_service.device)
		logger.info('No GPU acceleration available, using CPU')

	db.close()

	socket_service.model_load_completed(ModelLoadCompletedResponse(id=id))

	return pipe
