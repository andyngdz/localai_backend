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


def move_to_device(pipe, device, log_prefix):
	"""
	Helper function to move a model to a device, trying to_empty() first with fallback to to()
	"""
	try:
		# Try using to_empty() first for meta tensors
		pipe = pipe.to_empty(device)
		logger.info(f'{log_prefix}, moved to {device} device using to_empty()')
	except (AttributeError, TypeError):
		# Fall back to regular to() if to_empty() is not available or fails
		pipe = pipe.to(device)
		logger.info(f'{log_prefix}, moved to {device} device using to()')
	return pipe


def model_loader(id: str):
	db = SessionLocal()

	logger.info(f'Loading model {id} to {device_service.device}')

	max_memory = MaxMemoryConfig(db).to_dict()
	logger.info(f'Max memory configuration: {max_memory}')

	feature_extractor = CLIPImageProcessor.from_pretrained(CLIP_IMAGE_PROCESSOR_MODEL)
	safety_checker_instance = StableDiffusionSafetyChecker.from_pretrained(SAFETY_CHECKER_MODEL)

	# Try multiple loading strategies to support various model formats
	loading_strategies = [
		# Strategy 1: FP16 safetensors (for models like Juggernaut XL)
		{
			'use_safetensors': True,
			'variant': 'fp16',
		},
		# Strategy 2: Standard safetensors
		{
			'use_safetensors': True,
		},
		# Strategy 3: FP16 without safetensors
		{
			'use_safetensors': False,
			'variant': 'fp16',
		},
		# Strategy 4: Standard without safetensors
		{
			'use_safetensors': False,
		},
	]

	pipe = None
	last_error = None

	for strategy_idx, strategy_params in enumerate(loading_strategies, 1):
		try:
			logger.info(f'Trying loading strategy {strategy_idx}/{len(loading_strategies)}: {strategy_params}')
			pipe = AutoPipelineForText2Image.from_pretrained(
				id,
				cache_dir=CACHE_FOLDER,
				low_cpu_mem_usage=True,
				max_memory=max_memory,
				torch_dtype=device_service.torch_dtype,
				safety_checker=safety_checker_instance,
				feature_extractor=feature_extractor,
				device_map='balanced',
				**strategy_params,
			)
			logger.info(f'Successfully loaded model using strategy {strategy_idx}')
			break
		except Exception as error:
			last_error = error
			logger.warning(f'Strategy {strategy_idx} failed: {error}')
			continue
			
	if pipe is None:
		error_msg = f'Failed to load model {id} with all strategies. Last error: {last_error}'
		logger.error(error_msg)
		socket_service.model_load_failed(ModelLoadFailed(id=id, error=str(last_error)))
		if last_error is not None:
			raise last_error
		else:
			raise RuntimeError(error_msg)

	# Reset device map to allow explicit device placement, then move pipeline
	if hasattr(pipe, 'reset_device_map'):
		pipe.reset_device_map()
		logger.info(f'Reset device map for pipeline {id}')
	
	# Move entire pipeline to target device using to_empty() for meta tensors
	pipe = move_to_device(pipe, device_service.device, f'Pipeline {id}')

	# Apply device-specific optimizations
	# Note: For the current models and library versions, device_map="balanced" handles device placement,
	# and CPU offloading is not supported. This limitation may not apply to all models or future library versions.
	if device_service.is_cuda:
		pipe.enable_attention_slicing()
		logger.info('Applied CUDA optimizations: attention slicing enabled, pipeline moved to GPU')
	elif device_service.is_mps:
		# For MPS, we can enable attention slicing
		pipe.enable_attention_slicing()
		logger.info('Applied MPS optimizations: attention slicing enabled, pipeline moved to MPS')
	else:
		# For CPU-only systems, just enable attention slicing for better memory usage
		pipe.enable_attention_slicing()
		logger.info('Applied CPU optimizations: attention slicing enabled, pipeline moved to CPU')

	db.close()

	socket_service.model_load_completed(ModelLoadCompletedResponse(id=id))

	return pipe
