import logging
import os
from pathlib import Path

from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor

from app.cores.constants.model_loader import (
	CLIP_IMAGE_PROCESSOR_MODEL,
	SAFETY_CHECKER_MODEL,
	ModelLoadingStrategy,
)
from app.cores.max_memory import MaxMemoryConfig
from app.database.service import SessionLocal
from app.services import device_service, storage_service
from app.socket import socket_service
from config import CACHE_FOLDER

from .schemas import ModelLoadCompletedResponse, ModelLoadFailed

logger = logging.getLogger(__name__)


def find_single_file_checkpoint(model_path: str) -> str | None:
	"""
	Detect single-file checkpoint (.safetensors) in the model directory.
	Returns the path to the checkpoint file if found, None otherwise.

	This handles community models from CivitAI and HuggingFace that use
	single-file checkpoints instead of the diffusers format.
	"""
	if not os.path.exists(model_path):
		return None

	# Look for .safetensors files in the root of the model directory
	checkpoint_files = list(Path(model_path).glob('*.safetensors'))

	if checkpoint_files:
		# Return the first checkpoint file found
		checkpoint_path = str(checkpoint_files[0])
		logger.info(f'Found single-file checkpoint: {checkpoint_path}')
		return checkpoint_path

	return None


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

	# Check if the model exists in cache and look for single-file checkpoints
	# HuggingFace cache structure: .cache/models--{org}--{model}/snapshots/{hash}/
	model_cache_path = storage_service.get_model_dir(id)
	checkpoint_path = None

	if os.path.exists(model_cache_path):
		# Look for the latest snapshot
		snapshots_dir = os.path.join(model_cache_path, 'snapshots')
		if os.path.exists(snapshots_dir):
			# Get the most recent snapshot directory
			snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
			if snapshots:
				# Use the first snapshot (could be improved to use the most recent one)
				latest_snapshot = os.path.join(snapshots_dir, snapshots[0])
				checkpoint_path = find_single_file_checkpoint(latest_snapshot)

	# Build loading strategies based on whether we found a single-file checkpoint
	loading_strategies = []

	# Strategy 0: Single-file checkpoint (highest priority for community models)
	if checkpoint_path:
		loading_strategies.append({
			'type': ModelLoadingStrategy.SINGLE_FILE,
			'checkpoint_path': checkpoint_path,
		})

	# Strategy 1: FP16 safetensors (diffusers format)
	loading_strategies.append({
		'type': ModelLoadingStrategy.PRETRAINED,
		'use_safetensors': True,
		'variant': 'fp16',
	})

	# Strategy 2: Standard safetensors (diffusers format)
	loading_strategies.append({
		'type': ModelLoadingStrategy.PRETRAINED,
		'use_safetensors': True,
	})

	# Strategy 3: FP16 without safetensors (diffusers format)
	loading_strategies.append({
		'type': ModelLoadingStrategy.PRETRAINED,
		'use_safetensors': False,
		'variant': 'fp16',
	})

	# Strategy 4: Standard without safetensors (diffusers format)
	loading_strategies.append({
		'type': ModelLoadingStrategy.PRETRAINED,
		'use_safetensors': False,
	})

	pipe = None
	last_error = None

	for strategy_idx, strategy_params in enumerate(loading_strategies, 1):
		try:
			strategy_type = strategy_params.get('type')
			logger.info(f'Trying loading strategy {strategy_idx}/{len(loading_strategies)} ({strategy_type}): {strategy_params}')

			if strategy_type == ModelLoadingStrategy.SINGLE_FILE:
				# Load from single-file checkpoint
				checkpoint = strategy_params['checkpoint_path']
				pipe = AutoPipelineForText2Image.from_single_file(
					checkpoint,
					cache_dir=CACHE_FOLDER,
					low_cpu_mem_usage=True,
					torch_dtype=device_service.torch_dtype,
					safety_checker=safety_checker_instance,
					feature_extractor=feature_extractor,
				)
			else:
				# Load from pretrained (diffusers format)
				# Create clean params dict without 'type' key for unpacking
				load_params = {k: v for k, v in strategy_params.items() if k != 'type'}
				pipe = AutoPipelineForText2Image.from_pretrained(
					id,
					cache_dir=CACHE_FOLDER,
					low_cpu_mem_usage=True,
					torch_dtype=device_service.torch_dtype,
					safety_checker=safety_checker_instance,
					feature_extractor=feature_extractor,
					**load_params,
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
		pipe.enable_vae_slicing()
		logger.info('Applied CUDA optimizations: attention slicing + VAE slicing enabled, pipeline moved to GPU')
	elif device_service.is_mps:
		# For MPS, we can enable attention slicing and VAE slicing
		pipe.enable_attention_slicing()
		pipe.enable_vae_slicing()
		logger.info('Applied MPS optimizations: attention slicing + VAE slicing enabled, pipeline moved to MPS')
	else:
		# For CPU-only systems, enable attention slicing and VAE slicing for better memory usage
		pipe.enable_attention_slicing()
		pipe.enable_vae_slicing()
		logger.info('Applied CPU optimizations: attention slicing + VAE slicing enabled, pipeline moved to CPU')

	db.close()

	socket_service.model_load_completed(ModelLoadCompletedResponse(id=id))

	return pipe
