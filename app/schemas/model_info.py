"""Model Info"""

from typing import Optional

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """
    Represents a Stable Diffusion model, either local or from Hugging Face Hub.
    """

    id: str = Field(
        default=...,
        description="Unique identifier for the model (Hugging Face repo ID).",
    )
    name: str = Field(default=..., description="Display name of the model.")
    author: Optional[str] = Field(
        default=None, description="Author or organization of the model on Hugging Face."
    )
    likes: Optional[int] = Field(
        default=None, description="Number of likes for the model on Hugging Face."
    )
    trending_score: Optional[float] = Field(
        default=None, description="Trending score of the model on Hugging Face."
    )
    downloads: Optional[int] = Field(
        default=0, description="Number of downloads for the model on Hugging Face."
    )
    tags: Optional[list[str]] = Field(
        default=[], description="List of tags associated with the model."
    )
    is_downloaded: bool = Field(
        default=False, description="True if the model is downloaded locally."
    )
    size_mb: Optional[float] = Field(
        default=None, description="Estimated size of the model in megabytes."
    )
    description: Optional[str] = Field(
        default=None, description="A brief description of the model."
    )


class ModelInfoListResponse(BaseModel):
    """
    Response model for listing Stable Diffusion models.
    """

    models_info: list[ModelInfo] = Field(
        default=..., description="List of Stable Diffusion models."
    )
