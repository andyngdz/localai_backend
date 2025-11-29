"""Traditional image upscaling with optional img2img refinement."""

from typing import cast

import torch
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from PIL import Image

from app.schemas.generators import GeneratorConfig, Img2ImgParams, OutputType
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
		images: list[Image.Image],
		generator: torch.Generator,
		scale_factor: float,
		upscaler_type: UpscalerType,
		hires_steps: int,
		denoising_strength: float,
	) -> list[Image.Image]:
		"""Upscale images using PIL interpolation and refine with img2img pass.

		Args:
			config: Generator config (also provides base steps if hires_steps=0)
			pipe: Diffusion pipeline
			images: Base PIL images to upscale
			generator: Torch generator for reproducibility
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
		refined = self.refine(config, pipe, upscaled, generator, actual_steps, denoising_strength)

		return refined

	def _upscale_pil(
		self,
		images: list[Image.Image],
		scale_factor: float,
		upscaler_type: UpscalerType,
	) -> list[Image.Image]:
		"""Upscale images using PIL interpolation."""
		original_width, original_height = images[0].size
		logger.info(
			f'Upscaling {len(images)} image(s) from {original_width}x{original_height} '
			f'by {scale_factor}x using {upscaler_type.value}'
		)

		resample_mode = upscaler_type.to_pil_resample()
		upscaled_images = []
		for img in images:
			new_width = int(img.width * scale_factor)
			new_height = int(img.height * scale_factor)
			upscaled_img = img.resize((new_width, new_height), resample=resample_mode)
			upscaled_images.append(upscaled_img)

		new_width, new_height = upscaled_images[0].size
		logger.info(f'Upscaled to {new_width}x{new_height}')

		return upscaled_images

	def refine(
		self,
		config: GeneratorConfig,
		pipe: DiffusersPipeline,
		images: list[Image.Image],
		generator: torch.Generator,
		steps: int,
		denoising_strength: float,
	) -> list[Image.Image]:
		"""Run img2img refinement pass to add detail and reduce upscaling blur.

		Args:
			config: Generator config (prompt, negative_prompt, cfg_scale, etc.)
			pipe: Diffusion pipeline
			images: Upscaled PIL images
			generator: Torch generator for reproducibility
			steps: Number of inference steps
			denoising_strength: How much to repaint (0=keep original, 1=fully repaint)

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


traditional_upscaler = TraditionalUpscaler()
