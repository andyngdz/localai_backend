"""Prompt processing and style application for image generation."""

from app.schemas.generators import GeneratorConfig
from app.services import logger_service, styles_service

logger = logger_service.get_logger(__name__, category='Generate')


class PromptProcessor:
	"""Handles prompt preparation and style application."""

	def prepare_prompts(self, config: GeneratorConfig) -> tuple[str, str]:
		"""Prepare positive and negative prompts by applying styles.

		Args:
			config: Generation configuration with prompt and styles

		Returns:
			Tuple of (positive_prompt, negative_prompt)
		"""
		positive_prompt, negative_prompt = styles_service.apply_styles(
			config.prompt,
			config.negative_prompt,
			config.styles,
		)

		logger.info(f'Positive prompt after clipping: {positive_prompt}')
		logger.info(f'Negative prompt after clipping: {negative_prompt}')

		return positive_prompt, negative_prompt


prompt_processor = PromptProcessor()
