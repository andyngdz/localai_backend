from pydantic import BaseModel, Field


class DownloadCompletedResponse(BaseModel):
    """
    Response model for a completed download.
    Contains the ID of the model that was downloaded.
    """

    id: str = Field(..., description='The ID of the model that was downloaded.')
