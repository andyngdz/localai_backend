import logging
import os
from pathlib import Path
from typing import Any, Optional, Union, cast

from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor

from app.cores.constants.model_loader import (
	CLIP_IMAGE_PROCESSOR_MODEL,
	SAFETY_CHECKER_MODEL,
	ModelLoadingStrategy,
)
from app.cores.gpu_utils import cleanup_gpu_model
from app.cores.max_memory import MaxMemoryConfig
from app.database.service import SessionLocal
from app.services import device_service, storage_service
from app.socket import socket_service
from config import CACHE_FOLDER

from .cancellation import CancellationException, CancellationToken
from .schemas import ModelLoadCompletedResponse, ModelLoadFailed, ModelLoadPhase, ModelLoadProgressResponse

logger = logging.getLogger(__name__)


def map_step_to_phase(step: int) -> ModelLoadPhase:
	"""Map checkpoint step to loading phase.

	Args:
		step: Current checkpoint number (1-9)

	Returns:
		Corresponding ModelLoadPhase
	"""
	if step <= 2:
		return ModelLoadPhase.INITIALIZATION
	elif step <= 5:
		return ModelLoadPhase.LOADING_MODEL
	elif step <= 7:
		return ModelLoadPhase.DEVICE_SETUP
	else:
		return ModelLoadPhase.OPTIMIZATION


def emit_progress(model_id: str, step: int, message: str) -> None:
	"""Emit model loading progress via WebSocket with structured logging.

	Args:
		model_id: ID of the model being loaded
		step: Current checkpoint number (1-9)
		message: Human-readable status message
	"""
	try:
		phase = map_step_to_phase(step)

		progress = ModelLoadProgressResponse(
			id=model_id,
			step=step,
			total=9,
			phase=phase,
			message=message,
		)

		# Structured logging for production observability
		logger.info(f'[ModelLoad] {model_id} step={step}/9 phase={phase.value} msg="{message}"')

		socket_service.model_load_progress(progress)
	except Exception as e:
		# Don't let progress emission failures interrupt model loading
		logger.warning(f'Failed to emit model load progress: {e}')


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


def find_checkpoint_in_cache(model_cache_path: str) -> str | None:
	"""
	Find a single-file checkpoint in the model cache directory.

	Searches through HuggingFace cache structure:
	.cache/models--{org}--{model}/snapshots/{hash}/

	Args:
		model_cache_path: Path to the model cache directory

	Returns:
		Path to checkpoint file if found, None otherwise
	"""
	if not os.path.exists(model_cache_path):
		return None

	# Look for the latest snapshot
	snapshots_dir = os.path.join(model_cache_path, 'snapshots')
	if not os.path.exists(snapshots_dir):
		return None

	# Get the most recent snapshot directory
	snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]

	if not snapshots:
		return None

	# Use the first snapshot (could be improved to use the most recent one)
	latest_snapshot = os.path.join(snapshots_dir, snapshots[0])
	return find_single_file_checkpoint(latest_snapshot)


def apply_device_optimizations(pipe) -> None:
	"""
	Apply device-specific optimizations to the pipeline.

	Enables attention slicing and VAE slicing for better memory usage
	on CUDA, MPS, and CPU devices.

	Args:
		pipe: The pipeline to optimize
	"""
	pipe.enable_attention_slicing()
	pipe.enable_vae_slicing()

	if device_service.is_cuda:
		logger.info('Applied CUDA optimizations: attention slicing + VAE slicing enabled, pipeline moved to GPU')
	elif device_service.is_mps:
		logger.info('Applied MPS optimizations: attention slicing + VAE slicing enabled, pipeline moved to MPS')
	else:
		logger.info('Applied CPU optimizations: attention slicing + VAE slicing enabled, pipeline moved to CPU')


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


def cleanup_partial_load(pipe) -> None:
	"""Clean up partially loaded model resources on cancellation.

	Args:
		pipe: The partially loaded pipeline to clean up
	"""
	if pipe is None:
		return

	logger.info('Cleaning up partially loaded model...')
	metrics = cleanup_gpu_model(pipe, name='partial pipeline')
	logger.info(
		f'Partial load cleanup complete: {metrics.time_ms:.1f}ms, '
		f'{metrics.objects_collected} objects collected' + (f', error: {metrics.error}' if metrics.error else '')
	)


def model_loader(id: str, cancel_token: Optional[CancellationToken] = None):
	"""Load a model with optional cancellation support.

	Args:
		id: Model identifier to load
		cancel_token: Optional cancellation token for aborting the load

	Returns:
		Loaded pipeline instance

	Raises:
		CancellationException: If loading is cancelled via cancel_token
	"""
	db = SessionLocal()
	# Initialize pipe to None for exception handler scope (lines 341, 348)
	# Without this, if an exception occurs before pipe is assigned (e.g., during
	# SessionLocal, MaxMemoryConfig, or early checkpoints), cleanup_partial_load(pipe)
	# in exception handlers would raise NameError
	pipe = None

	try:
		logger.info(f'Loading model {id} to {device_service.device}')

		# Emit start event for frontend lifecycle management
		socket_service.model_load_started(ModelLoadCompletedResponse(id=id))

		# Checkpoint 1: Before initialization
		if cancel_token:
			cancel_token.check_cancelled()
		emit_progress(id, 1, 'Initializing model loader...')

		max_memory = MaxMemoryConfig(db).to_dict()
		logger.info(f'Max memory configuration: {max_memory}')

		# Checkpoint 2: Before loading feature extractor
		if cancel_token:
			cancel_token.check_cancelled()
		emit_progress(id, 2, 'Loading feature extractor...')

		feature_extractor = CLIPImageProcessor.from_pretrained(CLIP_IMAGE_PROCESSOR_MODEL)
		safety_checker_instance = StableDiffusionSafetyChecker.from_pretrained(SAFETY_CHECKER_MODEL)

		# Checkpoint 3: Before cache lookup
		if cancel_token:
			cancel_token.check_cancelled()
		emit_progress(id, 3, 'Checking model cache...')

		# Check if the model exists in cache and look for single-file checkpoints
		model_cache_path = storage_service.get_model_dir(id)
		checkpoint_path = find_checkpoint_in_cache(model_cache_path)

		# Checkpoint 4: Before building strategies
		if cancel_token:
			cancel_token.check_cancelled()
		emit_progress(id, 4, 'Preparing loading strategies...')

		# Build loading strategies based on whether we found a single-file checkpoint
		loading_strategies: list[dict[str, Any]] = []

		# Strategy 0: Single-file checkpoint (highest priority for community models)
		if checkpoint_path:
			loading_strategies.append(
				{
					'type': ModelLoadingStrategy.SINGLE_FILE,
					'checkpoint_path': checkpoint_path,
				}
			)

		# Strategy 1: FP16 safetensors (diffusers format)
		loading_strategies.append(
			{
				'type': ModelLoadingStrategy.PRETRAINED,
				'use_safetensors': True,
				'variant': 'fp16',
			}
		)

		# Strategy 2: Standard safetensors (diffusers format)
		loading_strategies.append(
			{
				'type': ModelLoadingStrategy.PRETRAINED,
				'use_safetensors': True,
			}
		)

		# Strategy 3: FP16 without safetensors (diffusers format)
		loading_strategies.append(
			{
				'type': ModelLoadingStrategy.PRETRAINED,
				'use_safetensors': False,
				'variant': 'fp16',
			}
		)

		# Strategy 4: Standard without safetensors (diffusers format)
		loading_strategies.append(
			{
				'type': ModelLoadingStrategy.PRETRAINED,
				'use_safetensors': False,
			}
		)

		last_error = None

		for strategy_idx, strategy_params in enumerate(loading_strategies, 1):
			# Checkpoint 5: Before each loading strategy
			if cancel_token:
				cancel_token.check_cancelled()
			emit_progress(id, 5, 'Loading model weights...')

			try:
				strategy_type = strategy_params.get('type')
				logger.info(
					f'Trying loading strategy {strategy_idx}/{len(loading_strategies)} ({strategy_type}): {strategy_params}'
				)

				if strategy_type == ModelLoadingStrategy.SINGLE_FILE:
					# Load from single-file checkpoint - try SDXL first, then SD 1.5
					from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
					from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline

					checkpoint = strategy_params['checkpoint_path']
					single_file_errors = []

					for pipeline_class in [StableDiffusionXLPipeline, StableDiffusionPipeline]:
						try:
							logger.debug(f'Trying {pipeline_class.__name__} for single-file checkpoint')
							# from_single_file exists at runtime but mypy type stubs don't expose it on the class type
							# Use getattr to call it and cast the result to the expected pipeline type
							from_single_file = getattr(pipeline_class, 'from_single_file')
							pipe = cast(
								Union[StableDiffusionXLPipeline, StableDiffusionPipeline],
								from_single_file(
									checkpoint,
									torch_dtype=device_service.torch_dtype,
								),
							)
							# Attach safety checker and feature extractor
							if hasattr(pipe, 'safety_checker'):
								pipe.safety_checker = safety_checker_instance
							if hasattr(pipe, 'feature_extractor'):
								pipe.feature_extractor = feature_extractor
							logger.info(f'Successfully loaded with {pipeline_class.__name__}')
							break
						except Exception as e:
							single_file_errors.append(f'{pipeline_class.__name__}: {e}')
							continue

					if not pipe:
						error_msg = f'Failed to load single-file checkpoint {checkpoint}. Tried: {", ".join(single_file_errors)}'
						raise ValueError(error_msg)
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

		# Checkpoint 6: After pipeline loaded, before device operations
		if cancel_token:
			cancel_token.check_cancelled()
		emit_progress(id, 6, 'Model loaded successfully')

		# Reset device map to allow explicit device placement, then move pipeline
		if hasattr(pipe, 'reset_device_map'):
			pipe.reset_device_map()
			logger.info(f'Reset device map for pipeline {id}')

		# Checkpoint 7: Before moving to device
		if cancel_token:
			cancel_token.check_cancelled()
		emit_progress(id, 7, 'Moving model to device...')

		# Move entire pipeline to target device using to_empty() for meta tensors
		pipe = move_to_device(pipe, device_service.device, f'Pipeline {id}')

		# Checkpoint 8: Before optimizations
		if cancel_token:
			cancel_token.check_cancelled()
		emit_progress(id, 8, 'Applying optimizations...')

		# Apply device-specific optimizations
		apply_device_optimizations(pipe)

		# Checkpoint 9: Before completion
		if cancel_token:
			cancel_token.check_cancelled()
		emit_progress(id, 9, 'Finalizing model setup...')

		db.close()

		socket_service.model_load_completed(ModelLoadCompletedResponse(id=id))

		return pipe

	except CancellationException:
		# Clean up partially loaded model on cancellation
		logger.info(f'Model loading cancelled for {id}, performing cleanup...')
		cleanup_partial_load(pipe)
		db.close()
		raise

	except Exception as e:
		# Clean up on any other error
		logger.error(f'Error loading model {id}: {e}')
		cleanup_partial_load(pipe)
		db.close()
		raise
