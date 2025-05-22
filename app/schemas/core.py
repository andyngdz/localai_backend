"""Core Models for the application."""

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """
    Error response schema.
    """

    detail: any = Field(..., description="Error message.")
    type: str = Field(..., description="Type of error.")
