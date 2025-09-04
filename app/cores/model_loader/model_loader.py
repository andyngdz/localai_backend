import logging

from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
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
			device_map='balanced',  # Use balanced device placement strategy
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
				device_map='balanced',  # Use balanced device placement strategy
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
	# Note: For the current models and library versions, device_map="balanced" handles device placement,
	# and CPU offloading is not supported. This limitation may not apply to all models or future library versions.
	if device_service.is_cuda:
		pipe.enable_attention_slicing()
		# Ensure safety checker is on the same device as the rest of the pipeline
		if hasattr(pipe, 'safety_checker') and pipe.safety_checker is not None:
			pipe.safety_checker = pipe.safety_checker.to(device_service.device)
		logger.info('Applied CUDA optimizations: attention slicing enabled, safety checker moved to GPU')
	elif device_service.is_mps:
		# For MPS, we can enable attention slicing
		pipe.enable_attention_slicing()
		# Ensure safety checker is on the same device as the rest of the pipeline
		if hasattr(pipe, 'safety_checker') and pipe.safety_checker is not None:
			pipe.safety_checker = pipe.safety_checker.to(device_service.device)
		logger.info('Applied MPS optimizations: attention slicing enabled, safety checker moved to MPS')
	else:
		# For CPU-only systems, just enable attention slicing for better memory usage
		pipe.enable_attention_slicing()
		logger.info('Applied CPU optimizations: attention slicing enabled')

	db.close()

	socket_service.model_load_completed(ModelLoadCompletedResponse(id=id))

	return pipe
