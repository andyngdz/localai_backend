"""Downloads model"""

from typing import Optional

from pydantic import BaseModel, Field


class DownloadRequest(BaseModel):
    """Request model for downloading"""

    id: str = Field(
        default=...,
        description='The Hugging Face repository ID of the model to download.',
    )
    hf_token: Optional[str] = Field(
        default=None,
        description='Optional Hugging Face API token for private models or increased rate limits.',
    )


class DownloadProgressResponse(BaseModel):
    """
    Progress model for tracking download status.
    """

    id: str = Field(default=..., description='The ID of the model being downloaded.')
    filename: str = Field(
        default=..., description='The name of the file currently being downloaded.'
    )
    downloaded: int = Field(
        default=0, description='The number of bytes downloaded so far.'
    )
    total: int = Field(
        default=0, description='The total size of the file being downloaded in bytes.'
    )


class DownloadStatusResponse(BaseModel):
    """
    Response schema for the status of a model download.
    """

    id: str = Field(default=..., description='The ID of the model being downloaded.')
    message: Optional[str] = Field(
        default=None, description='A human-readable message about the download status.'
    )
