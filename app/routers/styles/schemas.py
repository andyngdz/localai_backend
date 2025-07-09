from typing import Optional

from pydantic import BaseModel, Field


class StyleSchema(BaseModel):
    """
    Schema for style prompt.
    """

    positive: str = Field(..., description='The positive prompt for the style')
    negative: Optional[str] = Field(
        default=None, description='The negative prompt for the style'
    )
    image: str = Field(..., description='The image URL for the style')
