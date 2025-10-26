import asyncio
from concurrent.futures import ThreadPoolExecutor

import torch
from PIL import Image

from app.cores.generation import image_processor, memory_manager, progress_callback, seed_manager
from app.cores.model_manager import model_manager
from app.cores.pipeline_converter import pipeline_converter
from app.services import image_service, logger_service, styles_service

from .constants import DEFAULT_NEGATIVE_PROMPT
from .schemas import (
	ImageGenerationItem,
	ImageGenerationResponse,
	Img2ImgConfig,
)

logger = logger_service.get_logger(__name__, category='Generate')


class Img2ImgService:
	"""Service for image-to-image generation."""

	def __init__(self):
		self.executor = ThreadPoolExecutor()

	async def generate_image_from_image(self, config: Img2ImgConfig):
		"""
		Generate images from an input image using img2img pipeline.

		Args:
			config: Img2img configuration with source image and parameters.

		Returns:
			ImageGenerationResponse with generated images.

		Raises:
			ValueError: If model not loaded or generation fails.
		"""
		logger.info(f'Received img2img request: prompt="{config.prompt}", strength={config.strength}')

		if model_manager.pipe is None:
			logger.warning('Attempted img2img generation, but no model is loaded.')
			raise ValueError('No model is currently loaded')

		# Convert pipeline to img2img mode
		pipe = pipeline_converter.convert_to_img2img(model_manager.pipe)
		model_manager.pipe = pipe

		# Clear CUDA cache before generation
		memory_manager.clear_cache()

		# Validate batch size
		memory_manager.validate_batch_size(config.number_of_images, config.width, config.height)

		try:
			# Decode source image from base64
			logger.info('Decoding source image from base64')
			init_image = image_service.from_base64(config.init_image)
			logger.info(f'Source image size: {init_image.size}')

			# Resize source image to target dimensions
			init_image = image_service.resize_image(init_image, config.width, config.height, config.resize_mode)
			logger.info(f'Resized source image to: {init_image.size}')

			logger.info(
				f'Generating img2img: prompt="{config.prompt}", '
				f'strength={config.strength}, steps={config.steps}, '
				f'CFG={config.cfg_scale}, size={config.width}x{config.height}'
			)

			# Set sampler
			model_manager.set_sampler(config.sampler)

			# Get seed
			random_seed = seed_manager.get_seed(config.seed)

			# Apply styles to prompts
			positive_prompt, negative_prompt = styles_service.apply_styles(
				config.prompt,
				config.styles,
			)
			final_positive_prompt = positive_prompt
			final_negative_prompt = negative_prompt or DEFAULT_NEGATIVE_PROMPT

			logger.info(f'Positive prompt: {final_positive_prompt}')
			logger.info(f'Negative prompt: {final_negative_prompt}')

			# Run img2img generation in thread pool
			logger.info('Starting img2img generation in separate thread')
			loop = asyncio.get_event_loop()

			output = await loop.run_in_executor(
				self.executor,
				lambda: pipe(
					prompt=final_positive_prompt,
					negative_prompt=final_negative_prompt,
					image=init_image,
					strength=config.strength,
					num_inference_steps=config.steps,
					guidance_scale=config.cfg_scale,
					generator=torch.Generator(device=pipe.device).manual_seed(random_seed),
					num_images_per_prompt=config.number_of_images,
					callback_on_step_end=progress_callback.callback_on_step_end,
					callback_on_step_end_tensor_inputs=['latents'],
				),
			)

			logger.info(f'Img2img generation completed: {output}')

			# Process results
			nsfw_content_detected = image_processor.is_nsfw_content_detected(output)
			generated_images = output.get('images', [])
			items: list[ImageGenerationItem] = []

			for image in generated_images:
				if isinstance(image, Image.Image):
					path, file_name = image_processor.save_image(image)
					items.append(ImageGenerationItem(path=path, file_name=file_name))

			return ImageGenerationResponse(
				items=items,
				nsfw_content_detected=nsfw_content_detected,
			)

		except ValueError:
			# Re-raise validation errors
			raise
		except torch.cuda.OutOfMemoryError as error:
			logger.error(f'OOM error during img2img: {error}')
			memory_manager.clear_cache()

			raise ValueError(
				f'Out of memory: {config.number_of_images} images at {config.width}x{config.height}. '
				f'Try: (1) Reduce to 1 image, (2) Lower resolution to 512x512, (3) Reduce strength, '
				f'or (4) Restart model.'
			)
		except Exception as error:
			logger.exception(f'Failed img2img for prompt: "{config.prompt}"')
			raise ValueError(f'Failed to generate img2img: {error}')
		finally:
			# Always clear cache
			memory_manager.clear_cache()


img2img_service = Img2ImgService()
