from typing import Optional

from pydantic import BaseModel, Field

from app.constants import SamplerType


class GeneratorConfig(BaseModel):
	"""Request model for generating an image."""

	cfg_scale: float = Field(
		7.5, ge=1, description='Classifier-Free Guidance scale (CFG scale).'
	)
	height: int = Field(512, ge=64, description='Height of the generated image.')
	width: int = Field(512, ge=64, description='Width of the generated image.')
	hires_fix: bool = Field(False, description='Enable high-resolution fix.')
	negative_prompt: Optional[str] = Field(
		..., max_length=1000, description='Negative prompt to avoid certain features.'
	)
	prompt: str = Field(
		..., max_length=1000, description='The text prompt for image generation.'
	)
	steps: int = Field(24, ge=1, description='Number of inference steps.')
	seed: int = Field(-1, description='Random seed for reproducibility.')
	sampler: SamplerType = Field(
		SamplerType.EULER_A,
		description='Sampler type for image generation.',
	)
	styles: list[str] = Field(
		default_factory=list,
		description='List of styles to apply to the generated image.',
	)


class ImageGenerationRequest(BaseModel):
	history_id: str = Field(
		..., description='History ID for tracking the generation process.'
	)
	id: str = Field(..., description='Socket ID for tracking the generation process.')
	config: GeneratorConfig = Field(
		..., description='Configuration for image generation.'
	)


class ImageGenerationEachStepResponse(BaseModel):
	id: str = Field(..., description='Socket ID for tracking the generation process.')
	image_base64: str = Field(
		..., description='Base64 encoded image generated at this step.'
	)
