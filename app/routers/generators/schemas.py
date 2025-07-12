from typing import Optional

from pydantic import BaseModel, Field

from app.model_manager import SamplerType


class SamplerItem(BaseModel):
    """Available sampler to send to the client."""

    name: str = Field(..., description='User-friendly name of the sampler.')
    value: str = Field(..., description='Internal enum value for the sampler.')
    description: Optional[str] = Field(
        None, description="Brief description of the sampler's characteristics."
    )


class GenerateImageRequest(BaseModel):
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


class GenerateImageResponse(BaseModel):
    """Response model for image generation (you might not need this if returning FileResponse directly)."""

    message: str = Field('Image generated successfully.', description='Status message.')
    path: str = Field(..., description='Path to the generated image file.')
