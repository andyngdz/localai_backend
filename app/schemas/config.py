from pydantic import BaseModel, Field


class UpscalerItem(BaseModel):
	"""An upscaler option with metadata."""

	value: str = Field(..., description='Internal enum value')
	name: str = Field(..., description='Display name')
	description: str = Field(..., description='Brief description')
	suggested_denoise_strength: float = Field(
		..., ge=0.0, le=1.0, description='Suggested denoise strength for composition preservation'
	)


class ConfigResponse(BaseModel):
	"""Complete config response."""

	upscalers: list[UpscalerItem] = Field(..., description='Available upscaler types')
