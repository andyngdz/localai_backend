from enum import Enum

from pydantic import BaseModel, Field


class ModelLoadPhase(str, Enum):
	"""
	Enum for the different phases of model loading.
	Using str as base class ensures it serializes to string in JSON.
	"""

	INITIALIZATION = 'initialization'
	LOADING_MODEL = 'loading_model'
	DEVICE_SETUP = 'device_setup'
	OPTIMIZATION = 'optimization'


class ModelLoadProgressResponse(BaseModel):
	"""Response model for model loading progress updates."""

	id: str = Field(..., description='The ID of the model being loaded.')
	step: int = Field(..., description='Current checkpoint (1-9).')
	total: int = Field(default=9, description='Total checkpoints.')
	phase: ModelLoadPhase = Field(..., description='Current loading phase.')
	message: str = Field(..., description='Human-readable progress message.')


class ModelLoadCompletedResponse(BaseModel):
	"""Response model for when a model has been successfully loaded."""

	id: str = Field(..., description='The ID of the model that was loaded.')


class ModelLoadFailed(BaseModel):
	"""Response model for when a model has failed to load."""

	id: str = Field(..., description='The ID of the model that failed to load.')
	error: str = Field(..., description='The error message.')


class ModelLoaderProgressStep(BaseModel):
	"""Progress step for model loader initialization."""

	id: int = Field(..., description='Step number.')
	message: str = Field(..., description='Progress message for this step.')
