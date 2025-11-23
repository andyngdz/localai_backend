from typing import Optional

from app.cores.gpu_utils import cleanup_gpu_model
from app.cores.platform_optimizations import get_optimizer
from app.schemas.model_loader import DiffusersPipeline
from app.services import device_service, logger_service

from .cancellation import CancellationToken
from .progress import emit_progress

logger = logger_service.get_logger(__name__, category='ModelLoad')


def apply_device_optimizations(pipe: DiffusersPipeline) -> None:
	optimizer = get_optimizer()
	optimizer.apply(pipe)
	logger.info(f'Applied {optimizer.get_platform_name()} optimizations successfully')


def move_to_device(pipe: DiffusersPipeline, device: str, log_prefix: str) -> DiffusersPipeline:
	try:
		pipe = pipe.to_empty(device)
		logger.info(f'{log_prefix}, moved to {device} device using to_empty()')
	except (AttributeError, TypeError):
		pipe = pipe.to(device)
		logger.info(f'{log_prefix}, moved to {device} device using to()')
	return pipe


def cleanup_partial_load(pipe: Optional[DiffusersPipeline]) -> None:
	if pipe is None:
		return

	logger.info('Cleaning up partially loaded model...')
	metrics = cleanup_gpu_model(pipe, name='partial pipeline')
	logger.info(
		f'Partial load cleanup complete: {metrics.time_ms:.1f}ms, '
		+ f'{metrics.objects_collected} objects collected'
		+ (f', error: {metrics.error}' if metrics.error else '')
	)


def finalize_model_setup(
	pipe: DiffusersPipeline,
	model_id: str,
	cancel_token: Optional[CancellationToken],
) -> DiffusersPipeline:
	if cancel_token:
		cancel_token.check_cancelled()

	emit_progress(model_id, 6, 'Model loaded successfully')

	if hasattr(pipe, 'reset_device_map'):
		pipe.reset_device_map()
		logger.info(f'Reset device map for pipeline {model_id}')

	if cancel_token:
		cancel_token.check_cancelled()

	emit_progress(model_id, 7, 'Moving model to device...')

	pipe = move_to_device(pipe, device_service.device, f'Pipeline {model_id}')

	if cancel_token:
		cancel_token.check_cancelled()

	emit_progress(model_id, 8, 'Applying optimizations...')
	apply_device_optimizations(pipe)

	if cancel_token:
		cancel_token.check_cancelled()

	emit_progress(model_id, 9, 'Finalizing model setup...')
	return pipe


__all__ = [
	'apply_device_optimizations',
	'move_to_device',
	'cleanup_partial_load',
	'finalize_model_setup',
]
