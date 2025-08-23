"""Downloads Feature Schemas"""

from typing import Optional

from pydantic import BaseModel, Field


class DownloadModelRequest(BaseModel):
	"""Request model for downloading"""

	id: str = Field(
		...,
		description='The Hugging Face repository ID of the model to download.',
	)
	hf_token: Optional[str] = Field(
		default=None,
		description='Optional Hugging Face API token for private models or increased rate limits.',
	)


class DownloadModelStartResponse(BaseModel):
	"""
	Response model for preparing a download.
	Contains the list of files to be downloaded.
	"""

	id: str = Field(..., description='The ID of the model being downloaded.')


class DownloadStepProgressResponse(BaseModel):
	"""
	Response model for a download step progress.
	"""

	id: str = Field(..., description='The ID of the model being downloaded.')
	step: int = Field(..., description='The current step of the download.')
	total: int = Field(..., description='The total number of steps in the download.')


class DownloadModelResponse(BaseModel):
	"""
	Response schema for the status of a model download.
	"""

	id: str = Field(..., description='The ID of the model being downloaded.')
	message: Optional[str] = Field(..., description='A human-readable message about the download status.')
