"""Hires fix processor for high-resolution image generation."""

import torch
from PIL import Image

from app.constants.upscalers import REALESRGAN_UPSCALERS
from app.cores.generation.realesrgan_upscaler import realesrgan_upscaler
from app.cores.generation.traditional_upscaler import traditional_upscaler
from app.schemas.generators import GeneratorConfig
from app.schemas.model_loader import DiffusersPipeline
from app.services import logger_service

logger = logger_service.get_logger(__name__, category='HiresFix')


class HiresFixProcessor:
	"""Orchestrates high-resolution fix for image generation.

	Routes to appropriate upscaler based on type:
	- AI upscalers (Real-ESRGAN): upscale only, no refinement needed
	- Traditional upscalers (Lanczos, etc.): upscale + img2img refinement
	"""

	def apply(
		self,
		config: GeneratorConfig,
		pipe: DiffusersPipeline,
		generator: torch.Generator,
		images: list[Image.Image],
	) -> list[Image.Image]:
		"""Apply hires fix to decoded base images.

		Args:
			config: Generator config (contains hires_fix, prompt, steps, etc.)
			pipe: Diffusion pipeline
			images: Decoded base PIL images
			generator: Torch generator for reproducibility

		Returns:
			Upscaled (and optionally refined) PIL images at higher resolution
		"""
		assert config.hires_fix is not None

		hires_config = config.hires_fix
		result: list[Image.Image] = []

		logger.info(f'Applying hires fix\n{logger_service.format_config(hires_config)}')

		if hires_config.upscaler in REALESRGAN_UPSCALERS:
			result = realesrgan_upscaler.upscale(
				images,
				hires_config.upscaler,
				hires_config.upscale_factor,
			)
		else:
			result = traditional_upscaler.upscale(
				config,
				pipe,
				images,
				generator,
				scale_factor=hires_config.upscale_factor,
				upscaler_type=hires_config.upscaler,
				hires_steps=hires_config.steps,
				denoising_strength=hires_config.denoising_strength,
			)

		logger.info('Hires fix completed')
		return result


hires_fix_processor = HiresFixProcessor()
