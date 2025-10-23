import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import torch
from PIL import Image

from app.cores.generation import image_processor, memory_manager, progress_callback, seed_manager
from app.cores.model_manager import model_manager
from app.services import styles_service

from .constants import DEFAULT_NEGATIVE_PROMPT
from .schemas import (
	GeneratorConfig,
	ImageGenerationItem,
	ImageGenerationResponse,
)

logger = logging.getLogger(__name__)


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

	async def generate_image(self, config: GeneratorConfig):
		"""Generate images from text prompts using text-to-image pipeline.

		Args:
			config: Generation configuration with prompt and parameters.

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

		# Validate batch size to prevent OOM errors
		memory_manager.validate_batch_size(config.number_of_images, config.width, config.height)

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
			positive_prompt, negative_prompt = styles_service.apply_styles(
				config.prompt,
				config.styles,
			)
			final_positive_prompt = positive_prompt
			final_negative_prompt = negative_prompt or DEFAULT_NEGATIVE_PROMPT

			logger.info(f'Positive prompt after clipping: {final_positive_prompt}')
			logger.info(f'Negative prompt after clipping: {final_negative_prompt}')

			# Run the image generation in a separate thread to avoid blocking
			# the event loop, especially for long-running tasks like image generation.
			# This is necessary because the pipeline may not be fully async-compatible.
			# Using ThreadPoolExecutor to run the blocking code in a separate thread.
			logger.info('Starting image generation in a separate thread.')

			loop = asyncio.get_event_loop()

			output = await loop.run_in_executor(
				self.executor,
				lambda: pipe(
					prompt=final_positive_prompt,
					negative_prompt=final_negative_prompt,
					num_inference_steps=config.steps,
					guidance_scale=config.cfg_scale,
					height=config.height,
					width=config.width,
					generator=torch.Generator(device=pipe.device).manual_seed(random_seed),
					num_images_per_prompt=config.number_of_images,
					callback_on_step_end=progress_callback.callback_on_step_end,
					callback_on_step_end_tensor_inputs=['latents'],
				),
			)

			logger.info(f'Image generation completed successfully: {output}')

			nsfw_content_detected = image_processor.is_nsfw_content_detected(output)

			generated_image = output.get('images', [])
			items: list[ImageGenerationItem] = []

			for image in generated_image:
				if isinstance(image, Image.Image):
					path, file_name = image_processor.save_image(image)
					items.append(ImageGenerationItem(path=path, file_name=file_name))

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
			# Always clear cache after generation to prevent memory buildup
			memory_manager.clear_cache()


generator_service = GeneratorService()
