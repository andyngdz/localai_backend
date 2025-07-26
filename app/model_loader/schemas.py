from pydantic import BaseModel, Field


class ModelLoadCompletedResponse(BaseModel):
	"""Response model for when a model has been successfully loaded."""

	id: str = Field(..., description='The ID of the model that was loaded.')
