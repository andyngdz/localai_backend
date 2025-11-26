from enum import Enum

from pydantic import BaseModel, Field


class InterpolationMode(str, Enum):
	"""Torch interpolation modes for upscaling."""

	BILINEAR = 'bilinear'
	NEAREST = 'nearest'
	NEAREST_EXACT = 'nearest-exact'


class UpscalerType(str, Enum):
	"""Upscaling method for high-resolution fix."""

	LATENT = 'Latent'
	LATENT_NEAREST = 'Latent (nearest)'
	LATENT_NEAREST_EXACT = 'Latent (nearest-exact)'


class HiresFixConfig(BaseModel):
	"""Configuration for high-resolution fix (two-pass generation).

	Hires fix generates at base resolution first, then upscales and refines
	with img2img to avoid artifacts at high resolutions.
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
