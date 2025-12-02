from enum import Enum

from pydantic import BaseModel, Field


class UpscalingMethod(str, Enum):
	"""Upscaling method type indicator."""

	TRADITIONAL = 'traditional'
	AI = 'ai'


class UpscalerItem(BaseModel):
	"""An upscaler option with metadata."""

	value: str = Field(..., description='Internal enum value')
	name: str = Field(..., description='Display name')
	description: str = Field(..., description='Brief description')
	suggested_denoise_strength: float = Field(
		..., ge=0.0, le=1.0, description='Suggested denoise strength for composition preservation'
	)
	method: UpscalingMethod = Field(..., description='Upscaling method type (traditional or AI)')
	is_recommended: bool = Field(..., description='Whether this upscaler is recommended for typical use')


class UpscalerSection(BaseModel):
	"""A section grouping upscalers by method."""

	method: UpscalingMethod = Field(..., description='Upscaling method type')
	title: str = Field(..., description='Display title for the section')
	options: list[UpscalerItem] = Field(..., description='Upscaler options in this section')


class ConfigResponse(BaseModel):
	"""Complete config response."""

	upscalers: list[UpscalerSection] = Field(..., description='Upscaler sections grouped by method')
