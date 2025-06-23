"""Downloads model"""

from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, Field


class DownloadStatusStates(str, Enum):
    """
    Enum for the state of a model download.
    Using str as base class ensures it serializes to string in JSON.
    """

    PENDING = 'pending'
    DOWNLOADING = 'downloading'
    COMPLETED = 'completed'
    FAILED = 'failed'


class HuggingFaceRequest(BaseModel):
    """Request model for downloading"""

    model_id: str = Field(
        default=...,
        description='The Hugging Face repository ID of the model to download.',
    )
    hf_token: Optional[str] = Field(
        default=None,
        description='Optional Hugging Face API token for private models or increased rate limits.',
    )


class DownloadStatus(BaseModel):
    """Model download status."""

    model_id: str = Field(
        default=..., description='The Hugging Face repository ID of the model.'
    )
    status: Optional[DownloadStatusStates] = Field(
        default=DownloadStatusStates.PENDING,
        description='The current status of the download.',
    )
    progress: Optional[float] = Field(
        default=0.0, description='The progress of the download as a percentage (0-100).'
    )
    message: Optional[str] = Field(
        default=None,
        description='A message providing additional information about the status.',
    )
    error: Optional[str] = Field(
        default=None, description='Error message if the download failed.'
    )


class DownloadStatusResponse(BaseModel):
    """
    Response schema for the status of a model download.
    """

    model_id: str = Field(
        default=..., description='The ID of the model being downloaded.'
    )
    status: DownloadStatusStates = Field(
        default=..., description='Current status of the download.'
    )
    progress: float = Field(
        default=0.0, description='Download progress percentage (0.0 to 100.0).'
    )
    current_file: Optional[str] = Field(
        default=None, description='Name of the file currently being downloaded.'
    )
    total_size_bytes: Optional[int] = Field(
        default=None, description='Total size of the current file in bytes.'
    )
    message: Optional[str] = Field(
        default=None, description='A human-readable message about the download status.'
    )
    error: Optional[str] = Field(
        default=None, description='Error message if the download failed.'
    )
    path: Optional[str] = Field(
        default=None, description='Local path where the model is stored if completed.'
    )


download_statuses: Dict[str, DownloadStatus] = {}
