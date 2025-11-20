import os
from pathlib import Path
from typing import Literal, NotRequired, Optional, TypedDict, Union, cast

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
from app.cores.model_manager.pipeline_manager import DiffusersPipeline
from app.cores.platform_optimizations import get_optimizer
from app.database.service import SessionLocal
from app.services import device_service, logger_service, storage_service
from app.socket import socket_service
from config import CACHE_FOLDER

from .cancellation import CancellationException, CancellationToken
from .schemas import ModelLoadCompletedResponse, ModelLoadFailed, ModelLoadPhase, ModelLoadProgressResponse

logger = logger_service.get_logger(__name__, category='ModelLoad')


class SingleFileStrategy(TypedDict):
	type: Literal[ModelLoadingStrategy.SINGLE_FILE]
	checkpoint_path: str


class PretrainedStrategy(TypedDict):
	type: Literal[ModelLoadingStrategy.PRETRAINED]
	use_safetensors: bool
	variant: NotRequired[str]


Strategy = Union[SingleFileStrategy, PretrainedStrategy]


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
		logger.info(f'{model_id} step={step}/9 phase={phase.value} msg="{message}"')

		socket_service.model_load_progress(progress)
	except Exception as e:
		# Don't let progress emission failures interrupt model loading
		logger.warning(f'Failed to emit model load progress: {e}')


def find_single_file_checkpoint(model_path: str) -> Optional[str]:
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


def find_checkpoint_in_cache(model_cache_path: str) -> Optional[str]:
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


def apply_device_optimizations(pipe: DiffusersPipeline) -> None:
	"""
	Apply platform-specific optimizations to the pipeline.

	Uses modular platform-specific optimizers for Windows, Linux, and macOS.
	Each platform has its own optimized configuration based on hardware characteristics.

	Args:
		pipe: The pipeline to optimize
	"""

	optimizer = get_optimizer()
	optimizer.apply(pipe)
	logger.info(f'Applied {optimizer.get_platform_name()} optimizations successfully')


def move_to_device(pipe: DiffusersPipeline, device: str, log_prefix: str) -> DiffusersPipeline:
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


def cleanup_partial_load(pipe: Optional[DiffusersPipeline]) -> None:
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
		+ f'{metrics.objects_collected} objects collected'
		+ (f', error: {metrics.error}' if metrics.error else '')
	)


def _build_loading_strategies(
	checkpoint_path: Optional[str],
) -> list[Strategy]:
	strategies: list[Strategy] = []

	if checkpoint_path:
		strategies.append(
			{
				'type': ModelLoadingStrategy.SINGLE_FILE,
				'checkpoint_path': checkpoint_path,
			}
		)

	strategies.append({'type': ModelLoadingStrategy.PRETRAINED, 'use_safetensors': True})
	strategies.append({'type': ModelLoadingStrategy.PRETRAINED, 'use_safetensors': False})
	strategies.append({'type': ModelLoadingStrategy.PRETRAINED, 'use_safetensors': True, 'variant': 'fp16'})
	strategies.append({'type': ModelLoadingStrategy.PRETRAINED, 'use_safetensors': False, 'variant': 'fp16'})

	return strategies


def _load_single_file(
	checkpoint: str,
	safety_checker: StableDiffusionSafetyChecker,
	feature_extractor: CLIPImageProcessor,
) -> DiffusersPipeline:
	from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
	from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline

	errors = []

	for pipeline_class in [StableDiffusionXLPipeline, StableDiffusionPipeline]:
		try:
			logger.debug(f'Trying {pipeline_class.__name__} for single-file checkpoint')

			from_single_file = getattr(pipeline_class, 'from_single_file')

			pipe = cast(
				DiffusersPipeline,
				from_single_file(
					checkpoint,
					torch_dtype=device_service.torch_dtype,
				),
			)

			if hasattr(pipe, 'safety_checker'):
				pipe.safety_checker = safety_checker

			if hasattr(pipe, 'feature_extractor'):
				pipe.feature_extractor = feature_extractor

			logger.info(f'Successfully loaded with {pipeline_class.__name__}')
			return pipe
		except Exception as e:
			errors.append(f'{pipeline_class.__name__}: {e}')

	raise ValueError(f'Failed to load single-file checkpoint {checkpoint}. Tried: {", ".join(errors)}')


def _load_pretrained(
	id: str,
	params: PretrainedStrategy,
	safety_checker: StableDiffusionSafetyChecker,
	feature_extractor: CLIPImageProcessor,
) -> DiffusersPipeline:
	load_params = {k: v for k, v in params.items() if k != 'type'}

	return AutoPipelineForText2Image.from_pretrained(
		id,
		cache_dir=CACHE_FOLDER,
		low_cpu_mem_usage=True,
		torch_dtype=device_service.torch_dtype,
		safety_checker=safety_checker,
		feature_extractor=feature_extractor,
		**load_params,
	)


def _get_strategy_type(strategy: Strategy) -> ModelLoadingStrategy:
	strategy_type = cast(ModelLoadingStrategy, strategy['type'])
	if strategy_type not in (
		ModelLoadingStrategy.SINGLE_FILE,
		ModelLoadingStrategy.PRETRAINED,
	):
		raise ValueError(f'Unsupported strategy type: {strategy_type}')

	return strategy_type


def _load_single_file_strategy(
	strategy: SingleFileStrategy,
	safety_checker: StableDiffusionSafetyChecker,
	feature_extractor: CLIPImageProcessor,
) -> DiffusersPipeline:
	checkpoint_path = strategy['checkpoint_path']
	if not checkpoint_path:
		raise ValueError('Missing checkpoint path for single-file strategy')

	return _load_single_file(checkpoint_path, safety_checker, feature_extractor)


def _load_pretrained_strategy(
	id: str,
	strategy: PretrainedStrategy,
	safety_checker: StableDiffusionSafetyChecker,
	feature_extractor: CLIPImageProcessor,
) -> DiffusersPipeline:
	return _load_pretrained(id, strategy, safety_checker, feature_extractor)


def _load_pipeline_from_strategy(
	id: str,
	strategy: Strategy,
	strategy_type: ModelLoadingStrategy,
	safety_checker: StableDiffusionSafetyChecker,
	feature_extractor: CLIPImageProcessor,
) -> DiffusersPipeline:
	if strategy_type == ModelLoadingStrategy.SINGLE_FILE:
		return _load_single_file_strategy(
			cast(SingleFileStrategy, strategy),
			safety_checker,
			feature_extractor,
		)

	if strategy_type == ModelLoadingStrategy.PRETRAINED:
		return _load_pretrained_strategy(
			id,
			cast(PretrainedStrategy, strategy),
			safety_checker,
			feature_extractor,
		)

	raise ValueError(f'Unsupported strategy type: {strategy_type}')


def _execute_loading_strategies(
	id: str,
	strategies: list[Strategy],
	safety_checker: StableDiffusionSafetyChecker,
	feature_extractor: CLIPImageProcessor,
	cancel_token: Optional[CancellationToken],
) -> DiffusersPipeline:
	last_error = None

	for idx, strategy in enumerate(strategies, 1):
		if cancel_token:
			cancel_token.check_cancelled()

		emit_progress(id, 5, 'Loading model weights...')

		try:
			strategy_type = _get_strategy_type(strategy)
			logger.info(f'Trying loading strategy {idx}/{len(strategies)} ({strategy_type}): {strategy}')

			pipe = _load_pipeline_from_strategy(
				id,
				strategy,
				strategy_type,
				safety_checker,
				feature_extractor,
			)

			logger.info(f'Successfully loaded model using strategy {idx}')
			return pipe
		except Exception as error:
			last_error = error
			logger.warning(f'Strategy {idx} failed: {error}')

			continue

	error_msg = f'Failed to load model {id} with all strategies. Last error: {last_error}'

	logger.error(error_msg)

	socket_service.model_load_failed(ModelLoadFailed(id=id, error=str(last_error)))
	if last_error is not None:
		raise last_error
	else:
		raise RuntimeError(error_msg)


def _finalize_model_setup(
	pipe: DiffusersPipeline, id: str, cancel_token: Optional[CancellationToken]
) -> DiffusersPipeline:
	if cancel_token:
		cancel_token.check_cancelled()

	emit_progress(id, 6, 'Model loaded successfully')

	if hasattr(pipe, 'reset_device_map'):
		pipe.reset_device_map()
		logger.info(f'Reset device map for pipeline {id}')

	if cancel_token:
		cancel_token.check_cancelled()

	emit_progress(id, 7, 'Moving model to device...')

	pipe = move_to_device(pipe, device_service.device, f'Pipeline {id}')

	if cancel_token:
		cancel_token.check_cancelled()

	emit_progress(id, 8, 'Applying optimizations...')

	apply_device_optimizations(pipe)

	if cancel_token:
		cancel_token.check_cancelled()

	emit_progress(id, 9, 'Finalizing model setup...')

	return pipe


def model_loader(id: str, cancel_token: Optional[CancellationToken] = None) -> DiffusersPipeline:
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
	pipe: Optional[DiffusersPipeline] = None

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

		strategies = _build_loading_strategies(checkpoint_path)

		pipe = _execute_loading_strategies(
			id,
			strategies,
			safety_checker_instance,
			feature_extractor,
			cancel_token,
		)

		pipe = _finalize_model_setup(pipe, id, cancel_token)

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
