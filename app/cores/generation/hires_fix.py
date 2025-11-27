"""Hires fix processor for high-resolution image generation."""

from typing import cast

import torch
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from PIL import Image

from app.cores.generation.upscaler import image_upscaler
from app.schemas.generators import GeneratorConfig, Img2ImgParams, OutputType
from app.schemas.model_loader import DiffusersPipeline
from app.services import logger_service

logger = logger_service.get_logger(__name__, category='HiresFix')


class HiresFixProcessor:
	"""Handles high-resolution fix for image generation.

	Upscales decoded PIL images in pixel space, then runs img2img refinement
	pass to add details and reduce upscaling blur.
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
			Refined PIL images at higher resolution
		"""
		assert config.hires_fix is not None

		hires_config = config.hires_fix

		logger.info(f'Applying hires fix\n{logger_service.format_config(hires_config)}')

		upscaled_images = image_upscaler.upscale(
			images,
			scale_factor=hires_config.upscale_factor,
			upscaler_type=hires_config.upscaler,
		)

		actual_steps = self._get_steps(hires_config.steps, config.steps)
		logger.info(f'Running refinement pass with {actual_steps} steps')

		refined_images = self._run_refinement(
			config,
			pipe,
			upscaled_images,
			generator,
			actual_steps,
			hires_config.denoising_strength,
		)

		logger.info('Hires fix completed')
		return refined_images

	def _get_steps(self, hires_steps: int, base_steps: int) -> int:
		"""Get actual steps for hires pass.

		Args:
			hires_steps: Configured hires steps (0 = use base)
			base_steps: Base generation steps

		Returns:
			Actual steps to use
		"""
		return hires_steps if hires_steps > 0 else base_steps

	def _run_refinement(
		self,
		config: GeneratorConfig,
		pipe: DiffusersPipeline,
		images: list[Image.Image],
		generator: torch.Generator,
		steps: int,
		denoising_strength: float,
	) -> list[Image.Image]:
		"""Run img2img refinement pass.

		Args:
			config: Generator config
			pipe: Diffusion pipeline
			images: Upscaled PIL images
			generator: Torch generator
			steps: Inference steps
			denoising_strength: How much to repaint (0-1)

		Returns:
			Refined PIL images
		"""
		batch_size = len(images)
		width, height = images[0].size

		params = Img2ImgParams(
			prompt=config.prompt,
			negative_prompt=config.negative_prompt,
			num_inference_steps=steps,
			guidance_scale=config.cfg_scale,
			generator=generator,
			clip_skip=config.clip_skip,
			output_type=OutputType.PIL,
			strength=denoising_strength,
			num_images_per_prompt=batch_size,
			height=height,
			width=width,
			image=images,
		)

		logger.info(f'Img2Img refinement\n{logger_service.format_config(params)}')

		output = cast(StableDiffusionPipelineOutput, pipe(**vars(params)))
		return cast(list[Image.Image], output.images)


hires_fix_processor = HiresFixProcessor()
