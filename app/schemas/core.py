"""Core Models for the application."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ErrorType(str, Enum):
    TypeError = "TypeError"
    ValueError = "ValueError"
    ValidationError = "ValidationError"
    FileNotFound = "FileNotFound"
    Unauthorized = "Unauthorized"
    InternalServerError = "InternalServerError"


class ErrorResponse(BaseModel):
    """
    Error response schema.
    """

    detail: Any = Field(default=..., description="Error message.")
    type: ErrorType = Field(default=..., description="Type of error.")
