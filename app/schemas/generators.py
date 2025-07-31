from typing import Optional

from pydantic import BaseModel, Field

from app.constants import SamplerType


class GeneratorConfig(BaseModel):
	"""Request model for generating an image."""

	width: int = Field(default=512, ge=64, description='Width of the generated image.')
	height: int = Field(default=512, ge=64, description='Height of the generated image.')
	hires_fix: bool = Field(default=False, description='Enable high-resolution fix.')
	number_of_images: int = Field(default=1, ge=1, description='Number of images to generate.')
	prompt: str = Field(..., max_length=1000, description='The text prompt for image generation.')
	negative_prompt: Optional[str] = Field(
		default=None, max_length=1000, description='Negative prompt to avoid certain features.'
	)
	cfg_scale: float = Field(default=7.5, ge=1, description='Classifier-Free Guidance scale (CFG scale).')
	steps: int = Field(default=24, ge=1, description='Number of inference steps.')
	seed: int = Field(default=-1, description='Random seed for reproducibility.')
	sampler: SamplerType = Field(
		default=SamplerType.EULER_A,
		description='Sampler type for image generation.',
	)
	styles: list[str] = Field(
		default_factory=list,
		description='List of styles to apply to the generated image.',
	)


class ImageGenerationRequest(BaseModel):
	history_id: int = Field(..., description='History ID for tracking the generation process.')
	config: GeneratorConfig = Field(..., description='Configuration for image generation.')


class ImageGenerationStepEndResponse(BaseModel):
	index: int = Field(..., description='Index of the step in the generation process.')
	current_step: int = Field(..., description='Current step number in the generation process.')
	timestep: float = Field(..., description='Current timestep in the generation process.')
	image_base64: str = Field(..., description='Base64 encoded image generated at this step.')


class ImageGenerationItem(BaseModel):
	path: str = Field(..., description='Path to the generated image file.')
	file_name: str = Field(..., description='Name of the generated image file.')


class ImageGenerationResponse(BaseModel):
	items: list[ImageGenerationItem] = Field(
		default_factory=list,
		description='List of images with their paths and file names.',
	)
	nsfw_content_detected: list[bool] = Field(
		default_factory=list, description='Indicates if the generated image is NSFW (Not Safe For Work).'
	)
