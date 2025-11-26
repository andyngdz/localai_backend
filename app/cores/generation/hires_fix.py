"""Hires fix processor for high-resolution image generation."""

from typing import cast

import torch
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput

from app.cores.generation.upscaler import latent_upscaler
from app.schemas.generators import GeneratorConfig, OutputType
from app.schemas.model_loader import DiffusersPipeline
from app.services import logger_service

logger = logger_service.get_logger(__name__, category='HiresFix')


class HiresFixProcessor:
	"""Handles high-resolution fix for image generation.

	Upscales base latents and runs img2img refinement pass to add details.
	"""

	def apply(
		self,
		config: GeneratorConfig,
		pipe: DiffusersPipeline,
		latents: torch.Tensor,
		generator: torch.Generator,
	) -> torch.Tensor:
		"""Apply hires fix to latents.

		Args:
			config: Generator config (contains hires_fix, prompt, steps, etc.)
			pipe: Diffusion pipeline
			latents: Base generation latents
			generator: Torch generator for reproducibility

		Returns:
			Refined latents at higher resolution
		"""
		assert config.hires_fix is not None

		hires_config = config.hires_fix

		logger.info(
			f'Applying hires fix: scale={hires_config.upscale_factor}x, '
			f'upscaler={hires_config.upscaler.value}, '
			f'denoising={hires_config.denoising_strength}, '
			f'steps={hires_config.steps}'
		)

		upscaled_latents = latent_upscaler.upscale(
			latents,
			scale_factor=hires_config.upscale_factor,
			upscaler_type=hires_config.upscaler,
		)

		actual_steps = self._get_steps(hires_config.steps, config.steps)
		logger.info(f'Running refinement pass with {actual_steps} steps')

		refined_latents = self._run_refinement(
			config,
			pipe,
			upscaled_latents,
			generator,
			actual_steps,
			hires_config.denoising_strength,
		)

		logger.info('Hires fix completed')
		return refined_latents

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
		latents: torch.Tensor,
		generator: torch.Generator,
		steps: int,
		denoising_strength: float,
	) -> torch.Tensor:
		"""Run img2img refinement pass.

		Args:
			config: Generator config
			pipe: Diffusion pipeline
			latents: Upscaled latents
			generator: Torch generator
			steps: Inference steps
			denoising_strength: How much to repaint (0-1)

		Returns:
			Refined latents
		"""
		params = {
			'prompt': config.prompt,
			'negative_prompt': config.negative_prompt,
			'num_inference_steps': steps,
			'strength': denoising_strength,
			'latents': latents,
			'generator': generator,
			'guidance_scale': config.cfg_scale,
			'clip_skip': config.clip_skip,
			'output_type': OutputType.LATENT,
		}

		output = cast(StableDiffusionPipelineOutput, pipe(**params))
		return cast(torch.Tensor, output.images)


hires_fix_processor = HiresFixProcessor()
