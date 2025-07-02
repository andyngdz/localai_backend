"""Model Info"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from app.database.models.model import Model


class ModelSearchInfo(BaseModel):
    """
    Represents a Stable Diffusion model, either local or from Hugging Face Hub.
    """

    id: str = Field(..., description='Hugging Face repo ID')
    author: Optional[str] = Field(None, description='Author or org')
    likes: Optional[int] = Field(None, description='Number of likes')
    trending_score: Optional[float] = Field(None, description='Trending score')
    downloads: int = Field(0, description='Downloads count')
    tags: list[str] = Field(default_factory=list, description='Tags')
    is_downloaded: bool = Field(False, description='Downloaded locally?')
    size_mb: Optional[float] = Field(None, description='Estimated model size (MB)')
    description: Optional[str] = Field(None, description='Model description')


class ModelSearchInfoListResponse(BaseModel):
    """
    Response model for listing Stable Diffusion models.
    """

    models_search_info: list[ModelSearchInfo] = Field(
        default_factory=list,
        description='List of Stable Diffusion models when searching.',
    )


class LoadModelResponse(BaseModel):
    """
    Response model for loading a Stable Diffusion model.
    """

    id: str = Field(
        default=...,
        description='Unique identifier for the model (Hugging Face repo ID).',
    )
    config: Dict[str, Any] = Field(
        default_factory=dict, description='Model configuration details.'
    )


class NewModelAvailableResponse(BaseModel):
    """
    Response model for notifying about a new model available.
    """

    id: str = Field(
        default=...,
        description='Unique identifier for the new model (Hugging Face repo ID).',
    )


class ModelAvailableResponse(BaseModel):
    """
    Response model for checking if a model is available.
    """

    id: str = Field(
        default=...,
        description='Unique identifier for the model (Hugging Face repo ID).',
    )
    is_downloaded: bool = Field(
        default=False, description='Is the model downloaded locally?'
    )


class ModelDownloadedResponse(BaseModel):
    """Return list of downloaded models."""

    models: list[Model] = Field(
        default_factory=list, description='List of downloaded models.'
    )

    class Config:
        arbitrary_types_allowed = True


class LoadModelRequest(BaseModel):
    """Request model for loading a model by ID."""

    id: str = Field(..., description='The ID of the model to load.')


class GenerateImageRequest(BaseModel):
    """Request model for generating an image."""

    prompt: str = Field(..., description='The text prompt for image generation.')
    num_inference_steps: Optional[int] = Field(
        30, ge=1, description='Number of inference steps.'
    )
    guidance_scale: Optional[float] = Field(
        7.5, ge=0.0, description='Guidance scale (CFG scale).'
    )
    seed: Optional[int] = Field(
        None, ge=0, description='Random seed for reproducibility.'
    )
    height: Optional[int] = Field(
        512, ge=64, description='Height of the generated image.'
    )
    width: Optional[int] = Field(
        512, ge=64, description='Width of the generated image.'
    )


class GenerateImageResponse(BaseModel):
    """Response model for image generation (you might not need this if returning FileResponse directly)."""

    message: str = Field('Image generated successfully.', description='Status message.')
    path: str = Field(..., description='Path to the generated image file.')


class AvailableSampler(BaseModel):
    """Available sampler to send to the client."""

    name: str = Field(..., description='User-friendly name of the sampler.')
    value: str = Field(..., description='Internal enum value for the sampler.')
    description: Optional[str] = Field(
        None, description="Brief description of the sampler's characteristics."
    )
