"""Configuration validation for image generation."""

from app.cores.generation import memory_manager
from app.cores.model_manager import model_manager
from app.schemas.generators import GeneratorConfig
from app.services import logger_service

logger = logger_service.get_logger(__name__, category='Generate')


class ConfigValidator:
	"""Validates generation configuration before execution."""

	def validate_config(self, config: GeneratorConfig) -> None:
		"""Validate generation configuration.

		Args:
			config: Generation configuration to validate

		Raises:
			ValueError: If model is not loaded or configuration is invalid
		"""
		# Check if model is loaded
		if model_manager.pipe is None:
			logger.warning('Attempted to generate image, but no model is loaded.')
			raise ValueError('No model is currently loaded')

		# Validate batch size to prevent OOM errors
		memory_manager.validate_batch_size(
			config.number_of_images,
			config.width,
			config.height,
		)


config_validator = ConfigValidator()
