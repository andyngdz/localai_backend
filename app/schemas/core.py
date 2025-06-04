"""Core Models for the application."""

from typing import Any
from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """
    Error response schema.
    """

    detail: Any = Field(default=..., description="Error message.")
    type: str = Field(default=..., description="Type of error.")
