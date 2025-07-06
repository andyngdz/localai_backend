from typing import Optional

from pydantic import BaseModel, Field

from app.services.model_manager.schedulers import SamplerType


class GenerateImageRequest(BaseModel):
    """Request model for generating an image."""

    batch_count: int = Field(
        4, ge=1, description='Number of images to generate in a batch.'
    )
    batch_size: int = Field(2, ge=1, description='Number of images per batch.')
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


class GenerateImageResponse(BaseModel):
    """Response model for image generation (you might not need this if returning FileResponse directly)."""

    message: str = Field('Image generated successfully.', description='Status message.')
    path: str = Field(..., description='Path to the generated image file.')
