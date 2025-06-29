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


class DownloadCompletedResponse(BaseModel):
    """
    Response model for a completed download.
    Contains the ID of the model that was downloaded.
    """

    id: str = Field(default=..., description='The ID of the model that was downloaded.')


class DownloadPrepareResponse(BaseModel):
    """
    Response model for preparing a download.
    Contains the list of files to be downloaded.
    """

    id: str = Field(default=..., description='The ID of the model being downloaded.')
    files: list[str] = Field(
        default=..., description='List of files to be downloaded for the model.'
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


class DownloadCancelledResponse(BaseModel):
    """Response model for cancelling a download."""

    id: str = Field(default=..., description='The ID of the model download to cancel.')


class DownloadStatusResponse(BaseModel):
    """
    Response schema for the status of a model download.
    """

    id: str = Field(default=..., description='The ID of the model being downloaded.')
    message: Optional[str] = Field(
        default=None, description='A human-readable message about the download status.'
    )
