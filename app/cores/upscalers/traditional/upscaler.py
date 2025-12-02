"""Traditional image upscaling with PIL interpolation."""

import torch
from PIL import Image

from app.cores.upscalers.traditional.refiner import img2img_refiner
from app.schemas.generators import GeneratorConfig
from app.schemas.hires_fix import UpscalerType
from app.schemas.model_loader import DiffusersPipeline
from app.services import logger_service

logger = logger_service.get_logger(__name__, category='Upscaler')


class TraditionalUpscaler:
	"""Handles traditional (PIL) upscaling with img2img refinement.

	For traditional upscalers (Lanczos, Bicubic, etc.), upscaling produces blurry results.
	The refinement pass uses img2img to add detail and reduce blur.
	"""

	def upscale(
		self,
		config: GeneratorConfig,
		pipe: DiffusersPipeline,
		generator: torch.Generator,
		images: list[Image.Image],
		scale_factor: float,
		upscaler_type: UpscalerType,
		hires_steps: int,
		denoising_strength: float,
	) -> list[Image.Image]:
		"""Upscale images using PIL interpolation and refine with img2img pass.

		Args:
			config: Generator config (also provides base steps if hires_steps=0)
			pipe: Diffusion pipeline
			generator: Torch generator for reproducibility
			images: Base PIL images to upscale
			scale_factor: Upscaling factor
			upscaler_type: PIL upscaling method
			hires_steps: Inference steps for refinement (0 = use config.steps)
			denoising_strength: How much to repaint during refinement

		Returns:
			Upscaled and refined PIL images
		"""
		if not images:
			return []

		upscaled = self._upscale_pil(images, scale_factor, upscaler_type)

		actual_steps = hires_steps if hires_steps > 0 else config.steps
		logger.info(f'Running refinement pass with {actual_steps} steps')
		refined = img2img_refiner.refine(config, pipe, generator, upscaled, actual_steps, denoising_strength)

		return refined

	def _upscale_pil(
		self,
		images: list[Image.Image],
		scale_factor: float,
		upscaler_type: UpscalerType,
	) -> list[Image.Image]:
		"""Upscale images using PIL interpolation."""
		original_width, original_height = images[0].size
		config = {
			'batch_size': len(images),
			'original_size': f'{original_width}x{original_height}',
			'scale_factor': scale_factor,
			'upscaler': upscaler_type.value,
		}
		logger.info(f'PIL upscaling\n{logger_service.format_config(config)}')

		resample_mode = upscaler_type.to_pil_resample()
		upscaled_images: list[Image.Image] = []

		for img in images:
			new_width = int(img.width * scale_factor)
			new_height = int(img.height * scale_factor)
			upscaled_img = img.resize((new_width, new_height), resample=resample_mode)
			upscaled_images.append(upscaled_img)

		new_width, new_height = upscaled_images[0].size
		logger.info(f'Upscaled to {new_width}x{new_height}')

		return upscaled_images


traditional_upscaler = TraditionalUpscaler()
