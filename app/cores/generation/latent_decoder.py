"""Latent decoding utilities using pipeline components."""

import numpy as np
import torch
from PIL import Image

from app.schemas.generators import OutputType
from app.schemas.model_loader import DiffusersPipeline
from app.services import logger_service

logger = logger_service.get_logger(__name__, category='Generate')


class LatentDecoder:
	"""Decodes latent tensors to PIL images using pipeline's VAE and safety checker."""

	def decode_latents(self, pipe: DiffusersPipeline, latents: torch.Tensor) -> list[Image.Image]:
		"""Decode latent tensor to PIL images using pipeline's VAE.

		Args:
			pipe: The diffusion pipeline (has vae, image_processor)
			latents: Latent tensor [B, 4, H, W]

		Returns:
			List of PIL images
		"""
		with torch.no_grad():
			scaled_latents = latents / pipe.vae.config.scaling_factor
			decoder_output = pipe.vae.decode(scaled_latents)
			result = pipe.image_processor.postprocess(decoder_output.sample, output_type=OutputType.PIL.value)

		# When output_type=PIL, postprocess returns list[Image.Image]
		assert isinstance(result, list)
		return result

	def run_safety_checker(
		self,
		pipe: DiffusersPipeline,
		images: list[Image.Image],
	) -> tuple[list[Image.Image], list[bool]]:
		"""Run pipeline's safety checker on images.

		Args:
			pipe: Pipeline with safety_checker and feature_extractor
			images: List of PIL images

		Returns:
			Tuple of (images, nsfw_detected)
			- Images may be blacked out if NSFW detected
			- nsfw_detected is list of bools per image
		"""
		if pipe.safety_checker is None or pipe.feature_extractor is None:
			logger.info('No safety checker available (likely SDXL)')
			return images, [False] * len(images)

		safety_checker_input = pipe.feature_extractor(images, return_tensors='pt').to(pipe.device)

		# Convert PIL to numpy (safety checker expects numpy arrays)
		numpy_images = np.stack([np.array(img) for img in images])

		checked_images_np, nsfw_detected = pipe.safety_checker(
			images=numpy_images,
			clip_input=safety_checker_input.pixel_values.to(pipe.dtype),
		)

		# Convert numpy back to PIL
		checked_images = [Image.fromarray(img) for img in checked_images_np]

		if any(nsfw_detected):
			nsfw_count = sum(nsfw_detected)
			logger.warning(f'NSFW content detected in {nsfw_count} of {len(nsfw_detected)} image(s)')
		else:
			logger.info('Safety checker: No NSFW content detected')

		return checked_images, nsfw_detected


latent_decoder = LatentDecoder()
