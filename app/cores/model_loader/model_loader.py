from typing import Optional

import pydash
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor

from app.constants.model_loader import (
	CLIP_IMAGE_PROCESSOR_MODEL,
	MODEL_LOADING_PROGRESS_STEPS,
	SAFETY_CHECKER_MODEL,
)
from app.cores.max_memory import MaxMemoryConfig
from app.database.service import SessionLocal
from app.schemas.model_loader import DiffusersPipeline, ModelLoadCompletedResponse
from app.services import device_service, logger_service, storage_service
from app.socket import socket_service

from .cancellation import CancellationException, CancellationToken
from .progress import emit_progress, map_step_to_phase
from .setup import (
	apply_device_optimizations,
	cleanup_partial_load,
	finalize_model_setup,
	move_to_device,
)
from .strategies import (
	AutoPipelineForText2Image,
	build_loading_strategies,
	execute_loading_strategies,
	find_checkpoint_in_cache,
	find_single_file_checkpoint,
)

logger = logger_service.get_logger(__name__, category='ModelLoad')


__all__ = [
	'model_loader',
	'emit_progress',
	'map_step_to_phase',
	'cleanup_partial_load',
	'finalize_model_setup',
	'apply_device_optimizations',
	'move_to_device',
	'build_loading_strategies',
	'execute_loading_strategies',
	'find_checkpoint_in_cache',
	'find_single_file_checkpoint',
	'AutoPipelineForText2Image',
]


def _emit_progress_step(
	model_id: str,
	step_id: int,
	cancel_token: Optional[CancellationToken],
) -> None:
	if cancel_token:
		cancel_token.check_cancelled()

	step = pydash.find(MODEL_LOADING_PROGRESS_STEPS, lambda entry: entry.id == step_id)
	if step:
		emit_progress(model_id, step_id, step.message)


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
		socket_service.model_load_started(ModelLoadCompletedResponse(model_id=id))

		# Checkpoint 1: Before initialization
		_emit_progress_step(id, 1, cancel_token)

		max_memory = MaxMemoryConfig(db).to_dict()
		logger.info(f'Max memory configuration: {max_memory}')

		# Checkpoint 2: Before loading feature extractor
		_emit_progress_step(id, 2, cancel_token)

		feature_extractor = CLIPImageProcessor.from_pretrained(CLIP_IMAGE_PROCESSOR_MODEL)

		safety_checker_instance = StableDiffusionSafetyChecker.from_pretrained(SAFETY_CHECKER_MODEL)

		# Checkpoint 3: Before cache lookup
		_emit_progress_step(id, 3, cancel_token)

		# Check if the model exists in cache and look for single-file checkpoints
		model_cache_path = storage_service.get_model_dir(id)

		checkpoint_path = find_checkpoint_in_cache(model_cache_path)

		# Checkpoint 4: Before building strategies
		_emit_progress_step(id, 4, cancel_token)

		strategies = build_loading_strategies(checkpoint_path)

		pipe = execute_loading_strategies(
			id,
			strategies,
			safety_checker_instance,
			feature_extractor,
			cancel_token,
		)

		pipe = finalize_model_setup(pipe, id, cancel_token)

		db.close()

		socket_service.model_load_completed(ModelLoadCompletedResponse(model_id=id))

		return pipe

	except CancellationException:
		# Clean up partially loaded model on cancellation
		logger.info(f'Model loading cancelled for {id}, performing cleanup...')
		cleanup_partial_load(pipe)
		db.close()
		raise

	except Exception as error:
		# Clean up on any other error
		logger.error(f'Error loading model {id}: {error}')
		cleanup_partial_load(pipe)
		db.close()
		raise
