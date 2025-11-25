from app.schemas.model_loader import ModelLoadPhase, ModelLoadProgressResponse
from app.services import logger_service
from app.socket import socket_service

logger = logger_service.get_logger(__name__, category='ModelLoad')


def map_step_to_phase(step: int) -> ModelLoadPhase:
	"""Map checkpoint step to loading phase."""

	if step <= 2:
		return ModelLoadPhase.INITIALIZATION
	if step <= 5:
		return ModelLoadPhase.LOADING_MODEL
	if step <= 7:
		return ModelLoadPhase.DEVICE_SETUP
	return ModelLoadPhase.OPTIMIZATION


def emit_progress(model_id: str, step: int, message: str) -> None:
	"""Emit model loading progress via WebSocket with structured logging."""

	try:
		phase = map_step_to_phase(step)

		progress = ModelLoadProgressResponse(
			model_id=model_id,
			step=step,
			total=9,
			phase=phase,
			message=message,
		)

		logger.info(f'{model_id} step={step}/9 phase={phase.value} msg="{message}"')
		socket_service.model_load_progress(progress)
	except Exception as error:  # pragma: no cover - protective logging
		logger.warning(f'Failed to emit model load progress: {error}')


__all__ = ['map_step_to_phase', 'emit_progress']
