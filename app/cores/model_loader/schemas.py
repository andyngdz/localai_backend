from pydantic import BaseModel, Field


class ModelLoadCompletedResponse(BaseModel):
	"""Response model for when a model has been successfully loaded."""

	id: str = Field(..., description='The ID of the model that was loaded.')


class ModelLoadFailed(BaseModel):
	"""Response model for when a model has failed to load."""

	id: str = Field(..., description='The ID of the model that failed to load.')
	error: str = Field(..., description='The error message.')
