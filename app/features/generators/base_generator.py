"""Core image generation logic."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import cast

import torch
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput

from app.cores.generation import progress_callback, seed_manager
from app.cores.generation.hires_fix import hires_fix_processor
from app.cores.generation.latent_decoder import latent_decoder
from app.cores.generation.safety_checker_service import safety_checker_service
from app.cores.model_manager import model_manager
from app.schemas.generators import GeneratorConfig, OutputType, Text2ImgParams
from app.schemas.model_loader import DiffusersPipeline
from app.services import logger_service

logger = logger_service.get_logger(__name__, category='Generate')


class BaseGenerator:
	"""Handles core pipeline execution for image generation."""

	def __init__(self, executor: ThreadPoolExecutor):
		"""Initialize generator with thread executor.

		Args:
			executor: ThreadPoolExecutor for async operations
		"""
		self.executor = executor

	async def execute_pipeline(
		self,
		config: GeneratorConfig,
		positive_prompt: str,
		negative_prompt: str,
	) -> StableDiffusionPipelineOutput:
		"""Execute the diffusion pipeline for image generation.

		Args:
			config: Generation configuration
			positive_prompt: Processed positive prompt
			negative_prompt: Processed negative prompt

		Returns:
			Pipeline output with generated images

		Raises:
			ValueError: If generation fails

		Note:
			Model validation is performed by GeneratorService before this method is called.
		"""
		pipe = model_manager.pipe

		logger.info(f"Generating: '{config.prompt}'\n{logger_service.format_config(config)}")

		# Set sampler
		model_manager.set_sampler(config.sampler)

		# Get seed for reproducibility
		random_seed = seed_manager.get_seed(config.seed)

		# Create generator for reproducibility
		generator = torch.Generator(device=pipe.device).manual_seed(random_seed)

		# Prepare pipeline parameters with type safety
		pipeline_params = Text2ImgParams(
			prompt=positive_prompt,
			negative_prompt=negative_prompt,
			num_inference_steps=config.steps,
			guidance_scale=config.cfg_scale,
			generator=generator,
			clip_skip=config.clip_skip,
			output_type=OutputType.LATENT,
			height=config.height,
			width=config.width,
			num_images_per_prompt=config.number_of_images,
			callback_on_step_end=progress_callback.callback_on_step_end,
			callback_on_step_end_tensor_inputs=['latents'],
		)

		logger.info('Starting image generation in a separate thread.')
		loop = asyncio.get_event_loop()

		output = await loop.run_in_executor(
			self.executor,
			lambda: pipe(**vars(pipeline_params)),
		)

		# When output_type='latent', the output.images contains latent tensors
		output_with_latents = cast(StableDiffusionPipelineOutput, output)
		base_latents = cast(torch.Tensor, output_with_latents.images)

		# Decode base latents to PIL images
		images = latent_decoder.decode_latents(pipe, base_latents)

		# Run safety checker on base resolution images
		# SafetyCheckerService handles: database config check, model load/unload, NSFW detection
		images, nsfw_detected = safety_checker_service.check_images(images)

		# Apply hires fix to safe images if configured
		if config.hires_fix:
			images = await self._apply_hires_fix(config, pipe, generator, images, nsfw_detected, loop)

		logger.info('Image generation completed successfully')

		return StableDiffusionPipelineOutput(
			images=images,
			nsfw_content_detected=nsfw_detected,
		)

	async def _apply_hires_fix(
		self,
		config: GeneratorConfig,
		pipe: DiffusersPipeline,
		generator: torch.Generator,
		images: list,
		nsfw_detected: list[bool],
		loop: asyncio.AbstractEventLoop,
	) -> list:
		"""Apply hires fix to safe images only.

		Args:
			config: Generator configuration
			pipe: Diffusion pipeline
			images: Decoded base images
			nsfw_detected: NSFW detection results for each image
			generator: Torch generator for reproducibility
			loop: Event loop for async execution

		Returns:
			Images with hires fix applied to safe ones
		"""
		safe_indices = [idx for idx, nsfw in enumerate(nsfw_detected) if not nsfw]

		if not safe_indices:
			logger.warning('All images flagged as NSFW, skipping hires fix')
			return images

		logger.info(f'Applying hires fix to {len(safe_indices)} safe image(s)')

		safe_images = [images[idx] for idx in safe_indices]

		hires_images = await loop.run_in_executor(
			self.executor,
			lambda: hires_fix_processor.apply(config, pipe, generator, safe_images),
		)

		for safe_idx, hires_img in zip(safe_indices, hires_images):
			images[safe_idx] = hires_img

		logger.info('Hires fix applied successfully')
		return images
