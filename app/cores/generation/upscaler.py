"""Image upscaling for hires fix."""

from PIL import Image

from app.schemas.hires_fix import UpscalerType
from app.services import logger_service

logger = logger_service.get_logger(__name__, category='Upscaler')

MIN_SCALE_FACTOR = 1.0


class ImageUpscaler:
	"""Handles upscaling of PIL images for hires fix.

	Upscales decoded PIL images in pixel space using standard interpolation.
	This preserves image quality by avoiding interpolation of latent tensors,
	which the VAE was never trained to decode.
	"""

	def upscale(
		self,
		images: list[Image.Image],
		scale_factor: float,
		upscaler_type: UpscalerType = UpscalerType.LANCZOS,
	) -> list[Image.Image]:
		"""Upscale PIL images using specified interpolation method.

		Args:
			images: List of PIL images to upscale
			scale_factor: Upscaling factor (e.g., 2.0 for 2x)
			upscaler_type: Upscaling method (Lanczos, Bicubic, Bilinear, Nearest)

		Returns:
			List of upscaled PIL images

		Raises:
			ValueError: If scale_factor <= 1.0
		"""
		if scale_factor <= MIN_SCALE_FACTOR:
			raise ValueError(f'scale_factor must be > {MIN_SCALE_FACTOR}, got {scale_factor}')

		if not images:
			return []

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


image_upscaler = ImageUpscaler()
