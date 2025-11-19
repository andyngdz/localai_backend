import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
from sqlalchemy.orm import Session

from app.cores.generation import image_processor, memory_manager, progress_callback, seed_manager
from app.cores.generation.image_utils import process_generated_images
from app.cores.model_manager import model_manager
from app.database import crud as database_service
from app.schemas.lora import LoRAData
from app.services import logger_service, styles_service

from .schemas import (
	GeneratorConfig,
	ImageGenerationItem,
	ImageGenerationResponse,
)

logger = logger_service.get_logger(__name__, category='Generate')


class GeneratorService:
	"""Service for text-to-image generation."""

	def __init__(self):
		self.executor = ThreadPoolExecutor()

	def apply_hires_fix(self, hires_fix: bool):
		"""Apply hires fix if requested (placeholder for future implementation)."""
		if hires_fix:
			logger.warning(
				'Hires fix requested, but not fully implemented in this MVP. Generating directly at requested resolution.'
			)

	def _load_loras_for_generation(self, config: GeneratorConfig, db: Session) -> bool:
		"""Load LoRAs for image generation if specified in config.

		Args:
			config: Generation configuration with LoRA settings
			db: Database session for loading LoRA information

		Returns:
			True if LoRAs were loaded, False otherwise

		Raises:
			ValueError: If a LoRA is not found in the database
		"""
		if not config.loras:
			return False

		logger.info(f'Loading {len(config.loras)} LoRAs for generation')
		lora_data: list[LoRAData] = []

		for lora_config in config.loras:
			lora = database_service.get_lora_by_id(db, lora_config.lora_id)
			if not lora:
				raise ValueError(f'LoRA with id {lora_config.lora_id} not found')

			lora_data.append(
				LoRAData(
					id=lora.id,
					name=lora.name,
					file_path=lora.file_path,
					weight=lora_config.weight,
				)
			)

		model_manager.pipeline_manager.load_loras(lora_data)
		return True

	def _prepare_prompts(self, config: GeneratorConfig) -> tuple[str, str]:
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

	def _process_generated_images(self, output: Any) -> tuple[list[ImageGenerationItem], list[bool]]:
		"""Process generated images and save them to disk.

		Args:
			output: Pipeline output containing generated images

		Returns:
			Tuple of (image_items, nsfw_content_detected)
		"""
		return process_generated_images(output)

	async def generate_image(self, config: GeneratorConfig, db: Session):
		"""Generate images from text prompts using text-to-image pipeline.

		Args:
			config: Generation configuration with prompt and parameters.
			db: Database session for loading LoRA information

		Returns:
			ImageGenerationResponse with generated images.

		Raises:
			ValueError: If model not loaded or generation fails.
		"""
		logger.info(f'Received image generation request: {config}')

		pipe = model_manager.pipe

		if pipe is None:
			logger.warning('Attempted to generate image, but no model is loaded.')
			raise ValueError('No model is currently loaded')

		# Clear CUDA cache before generation to maximize available memory
		memory_manager.clear_cache()

		# Reset progress callback state for new generation
		if hasattr(progress_callback, 'reset'):
			progress_callback.reset()

		# Validate batch size to prevent OOM errors
		memory_manager.validate_batch_size(config.number_of_images, config.width, config.height)

		# Load LoRAs if specified
		loras_loaded = self._load_loras_for_generation(config, db)

		try:
			logger.info(
				f"Generating image(s) for prompt: '{config.prompt}' "
				f'with steps={config.steps}, CFG={config.cfg_scale}, '
				f'size={config.width}x{config.height}, batch={config.number_of_images}'
			)

			model_manager.set_sampler(config.sampler)

			self.apply_hires_fix(config.hires_fix)

			random_seed = seed_manager.get_seed(config.seed)

			# Apply styles to the prompt
			final_positive_prompt, final_negative_prompt = self._prepare_prompts(config)

			# Run the image generation in a separate thread to avoid blocking
			# the event loop, especially for long-running tasks like image generation.
			# This is necessary because the pipeline may not be fully async-compatible.
			# Using ThreadPoolExecutor to run the blocking code in a separate thread.
			logger.info('Starting image generation in a separate thread.')

			loop = asyncio.get_event_loop()

			# Prepare pipeline parameters
			pipeline_params = {
				'prompt': final_positive_prompt,
				'negative_prompt': final_negative_prompt,
				'num_inference_steps': config.steps,
				'guidance_scale': config.cfg_scale,
				'height': config.height,
				'width': config.width,
				'generator': torch.Generator(device=pipe.device).manual_seed(random_seed),
				'num_images_per_prompt': config.number_of_images,
				'callback_on_step_end': progress_callback.callback_on_step_end,
				'callback_on_step_end_tensor_inputs': ['latents'],
				'clip_skip': config.clip_skip,
			}

			# Run image generation
			output = await loop.run_in_executor(
				self.executor,
				lambda: pipe(**pipeline_params),
			)

			logger.info(f'Image generation completed successfully: {output}')

			# Process generated images and save them to disk
			items, nsfw_content_detected = self._process_generated_images(output)

			# Final cleanup of the output dictionary
			del output
			memory_manager.clear_cache()

			return ImageGenerationResponse(
				items=items,
				nsfw_content_detected=nsfw_content_detected,
			)

		except FileNotFoundError as error:
			logger.error(f'Model directory not found: {error}')
			raise ValueError(f'Model files not found: {error}')
		except torch.cuda.OutOfMemoryError as error:
			logger.error(f'Out of memory error during image generation: {error}')
			# Clear cache to recover from OOM
			memory_manager.clear_cache()

			# Clear tensor cache if available
			if hasattr(image_processor, 'clear_tensor_cache'):
				image_processor.clear_tensor_cache()

			raise ValueError(
				f'Out of memory error: tried to allocate memory but GPU is full. '
				f'Current batch: {config.number_of_images} images at {config.width}x{config.height}. '
				f'Try: (1) Reduce number of images to 1-2, (2) Lower resolution to 512x512, '
				f'or (3) Restart the model to clear memory.'
			)
		except Exception as error:
			logger.exception(f'Failed to generate image for prompt: "{config.prompt}"')
			raise ValueError(f'Failed to generate image: {error}')
		finally:
			# Unload LoRAs if they were loaded
			if loras_loaded:
				try:
					model_manager.pipeline_manager.unload_loras()
				except Exception as error:
					logger.error(f'Failed to unload LoRAs: {error}')

			# Final safety cleanup
			memory_manager.clear_cache()

			# Reset callback state
			if hasattr(progress_callback, 'reset'):
				progress_callback.reset()


generator_service = GeneratorService()
