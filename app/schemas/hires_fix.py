from enum import Enum

from PIL import Image
from pydantic import BaseModel, Field


class UpscalerType(str, Enum):
	"""Image upscaling methods for high-resolution fix (pixel-space).

	These methods upscale decoded PIL images, not latent tensors.
	This preserves image quality by avoiding interpolation of latent space.
	"""

	LANCZOS = 'Lanczos'
	BICUBIC = 'Bicubic'
	BILINEAR = 'Bilinear'
	NEAREST = 'Nearest'

	def to_pil_resample(self) -> Image.Resampling:
		"""Convert to PIL resampling mode."""
		mapping = {
			UpscalerType.LANCZOS: Image.Resampling.LANCZOS,
			UpscalerType.BICUBIC: Image.Resampling.BICUBIC,
			UpscalerType.BILINEAR: Image.Resampling.BILINEAR,
			UpscalerType.NEAREST: Image.Resampling.NEAREST,
		}
		return mapping[self]


class HiresFixConfig(BaseModel):
	"""Configuration for high-resolution fix (two-pass generation).

	Hires fix generates at base resolution first, then upscales in pixel space,
	and refines with img2img to add details and reduce upscaling blur.
	"""

	upscale_factor: float = Field(
		...,
		ge=1.5,
		le=4.0,
		description='Factor to upscale the image by (e.g., 2.0 means 512x512 → 1024x1024).',
	)
	upscaler: UpscalerType = Field(
		...,
		description='Upscaling method to use.',
	)
	denoising_strength: float = Field(
		...,
		ge=0.0,
		le=1.0,
		description='How much to repaint during refinement (0=keep original, 1=fully repaint).',
	)
	steps: int = Field(
		...,
		ge=0,
		description='Inference steps for hires pass. 0 means use same as main generation.',
	)
