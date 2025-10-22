from typing import Optional

from pydantic import BaseModel, Field

from app.cores.samplers import SamplerType
from app.features.generators.schemas import (
	ImageGenerationItem,
	ImageGenerationResponse,
	ImageGenerationStepEndResponse,
)


class Img2ImgConfig(BaseModel):
	"""Configuration for image-to-image generation."""

	# Image-specific parameters
	init_image: str = Field(..., description='Base64 encoded source image.')
	strength: float = Field(
		default=0.75, ge=0.0, le=1.0, description='Strength of transformation (0=no change, 1=complete change).'
	)
	resize_mode: str = Field(
		default='resize', description='How to handle image size: "resize" (stretch) or "crop" (center crop).'
	)

	# Generation parameters (same as text-to-image)
	width: int = Field(default=512, ge=64, description='Width of the generated image.')
	height: int = Field(default=512, ge=64, description='Height of the generated image.')
	number_of_images: int = Field(default=1, ge=1, description='Number of images to generate.')
	prompt: str = Field(..., max_length=1000, description='The text prompt for image generation.')
	negative_prompt: Optional[str] = Field(
		default=None, max_length=1000, description='Negative prompt to avoid certain features.'
	)
	cfg_scale: float = Field(default=7.5, ge=1, description='Classifier-Free Guidance scale.')
	steps: int = Field(default=24, ge=1, description='Number of inference steps.')
	seed: int = Field(default=-1, description='Random seed for reproducibility.')
	sampler: SamplerType = Field(default=SamplerType.EULER_A, description='Sampler type for image generation.')
	styles: list[str] = Field(default_factory=list, description='List of styles to apply.')


class Img2ImgRequest(BaseModel):
	history_id: int = Field(..., description='History ID for tracking the generation process.')
	config: Img2ImgConfig = Field(..., description='Configuration for img2img generation.')


__all__ = [
	'Img2ImgConfig',
	'Img2ImgRequest',
	'ImageGenerationItem',
	'ImageGenerationResponse',
	'ImageGenerationStepEndResponse',
]
