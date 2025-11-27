from app.schemas.config import UpscalerItem
from app.schemas.hires_fix import UpscalerType

UPSCALER_METADATA: dict[UpscalerType, UpscalerItem] = {
	UpscalerType.LANCZOS: UpscalerItem(
		value='Lanczos',
		name='Lanczos (High Quality)',
		description='High-quality resampling, best for photos',
		suggested_denoise_strength=0.4,
	),
	UpscalerType.BICUBIC: UpscalerItem(
		value='Bicubic',
		name='Bicubic (Smooth)',
		description='Smooth interpolation, good balance',
		suggested_denoise_strength=0.4,
	),
	UpscalerType.BILINEAR: UpscalerItem(
		value='Bilinear',
		name='Bilinear (Fast)',
		description='Fast interpolation, moderate quality',
		suggested_denoise_strength=0.35,
	),
	UpscalerType.NEAREST: UpscalerItem(
		value='Nearest',
		name='Nearest (Sharp Edges)',
		description='No interpolation, preserves sharp edges',
		suggested_denoise_strength=0.3,
	),
}


class ConfigService:
	def get_upscalers(self) -> list[UpscalerItem]:
		return [UPSCALER_METADATA[upscaler] for upscaler in UpscalerType]


config_service = ConfigService()
