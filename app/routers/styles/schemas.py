from pydantic import BaseModel, Field

from app.predefined_styles.schemas import StyleItem


class StylesSectionResponse(BaseModel):
	"""
	Response model for styles section.
	"""

	id: str = Field(..., description='Unique identifier for the styles response')
	styles: list[StyleItem] = Field(
		default=[],
		description='List of style',
	)
