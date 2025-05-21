from pydantic import BaseModel, Field
from typing import Optional


class ModelInfo(BaseModel):
    """
    Represents a Stable Diffusion model, either local or from Hugging Face Hub.
    """

    id: str = Field(
        ..., description="Unique identifier for the model (Hugging Face repo ID)."
    )
    name: str = Field(..., description="Display name of the model.")
    author: Optional[str] = Field(
        None, description="Author or organization of the model on Hugging Face."
    )
    likes: Optional[int] = (
        Field(None, description="Number of likes for the model on Hugging Face."),
    )
    trending_score: Optional[float] = (
        Field(None, description="Trending score of the model on Hugging Face."),
    )
    downloads: Optional[int] = Field(
        None, description="Number of downloads for the model on Hugging Face."
    )
    tags: Optional[list[str]] = Field(
        [], description="List of tags associated with the model."
    )
    is_downloaded: bool = Field(
        False, description="True if the model is downloaded locally."
    )
    size_mb: Optional[float] = Field(
        None, description="Estimated size of the model in megabytes."
    )
    description: Optional[str] = Field(
        None, description="A brief description of the model."
    )


class ModelInfoListResponse(BaseModel):
    """
    Response model for listing Stable Diffusion models.
    """

    models_info: list[ModelInfo] = Field(
        ..., description="List of Stable Diffusion models."
    )
