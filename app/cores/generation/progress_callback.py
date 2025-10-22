"""Progress callback handler for step-by-step generation updates."""

import logging

from app.cores.generation.image_processor import image_processor
from app.services import image_service
from app.socket import socket_service

logger = logging.getLogger(__name__)


class ProgressCallback:
	"""Handles step-by-step progress callbacks via WebSocket."""

	def __init__(self):
		self.image_processor = image_processor

	def callback_on_step_end(self, pipe, current_step: int, timestep: float, callback_kwargs: dict) -> dict:
		"""Callback for step-by-step progress updates via WebSocket.

		Args:
			pipe: The diffusion pipeline instance.
			current_step: Current inference step number.
			timestep: Current timestep value.
			callback_kwargs: Dictionary containing latents and other callback data.

		Returns:
			The callback_kwargs dictionary (required by diffusers API).
		"""
		logger.info(f'Callback on step end: current_step={current_step}, timestep={timestep}')

		latents = callback_kwargs['latents']

		# Import here to avoid circular dependency
		from app.features.generators.schemas import ImageGenerationStepEndResponse

		for index, latent in enumerate(latents):
			image = self.image_processor.latents_to_rgb(latent)
			image_base64 = image_service.to_base64(image)

			logger.info(f'Generated preview for step {current_step}, index {index}')

			socket_service.image_generation_step_end(
				ImageGenerationStepEndResponse(
					current_step=current_step,
					image_base64=image_base64,
					index=index,
					timestep=timestep,
				)
			)

		return callback_kwargs


progress_callback = ProgressCallback()
