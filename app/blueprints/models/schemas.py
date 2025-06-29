"""Model Info"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from app.database.model import Model


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
        default=[], description='List of Stable Diffusion models when searching.'
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
        default={}, description='Model configuration details.'
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


class ModelDownloadedResponse(BaseModel, arbitrary_types_allowed=True):
    """Return list of downloaded models."""

    models: list[Model] = Field([], description='List of downloaded models.')
