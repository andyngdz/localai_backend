from typing import Optional

from pydantic import BaseModel, Field


class StyleSchema(BaseModel):
    """
    Schema for style prompt.
    """

    id: str = Field(..., description='The unique identifier for the style')
    name: str = Field(..., description='The name of the style')
    origin: Optional[str] = Field(default=None, description='The origin of the style')
    license: Optional[str] = Field(
        default='MIT', description='The license of the style'
    )
    positive: Optional[str] = Field(
        default='{prompt}', description='The positive prompt for the style'
    )
    negative: Optional[str] = Field(
        default=None, description='The negative prompt for the style'
    )
    image: str = Field(..., description='The image URL for the style')
