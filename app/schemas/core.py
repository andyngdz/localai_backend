"""Core Models for the application."""

from typing import Any
from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """
    Error response schema.
    """

    detail: Any = Field(..., description="Error message.")
    type: str = Field(..., description="Type of error.")
