from app.schemas.config import UpscalerItem, UpscalingMethod
from app.schemas.hires_fix import UpscalerType

UPSCALER_METADATA: dict[UpscalerType, UpscalerItem] = {
	UpscalerType.LANCZOS: UpscalerItem(
		value='Lanczos',
		name='Lanczos (High Quality)',
		description='High-quality resampling, best for photos',
		suggested_denoise_strength=0.4,
		method=UpscalingMethod.TRADITIONAL,
	),
	UpscalerType.BICUBIC: UpscalerItem(
		value='Bicubic',
		name='Bicubic (Smooth)',
		description='Smooth interpolation, good balance',
		suggested_denoise_strength=0.4,
		method=UpscalingMethod.TRADITIONAL,
	),
	UpscalerType.BILINEAR: UpscalerItem(
		value='Bilinear',
		name='Bilinear (Fast)',
		description='Fast interpolation, moderate quality',
		suggested_denoise_strength=0.35,
		method=UpscalingMethod.TRADITIONAL,
	),
	UpscalerType.NEAREST: UpscalerItem(
		value='Nearest',
		name='Nearest (Sharp Edges)',
		description='No interpolation, preserves sharp edges',
		suggested_denoise_strength=0.3,
		method=UpscalingMethod.TRADITIONAL,
	),
	UpscalerType.REALESRGAN_X2PLUS: UpscalerItem(
		value='RealESRGAN_x2plus',
		name='Real-ESRGAN 2x (General)',
		description='AI upscaler for general images, 2x native scale',
		suggested_denoise_strength=0.35,
		method=UpscalingMethod.AI,
	),
	UpscalerType.REALESRGAN_X4PLUS: UpscalerItem(
		value='RealESRGAN_x4plus',
		name='Real-ESRGAN 4x (General)',
		description='AI upscaler for general images, 4x native scale',
		suggested_denoise_strength=0.3,
		method=UpscalingMethod.AI,
	),
	UpscalerType.REALESRGAN_X4PLUS_ANIME: UpscalerItem(
		value='RealESRGAN_x4plus_anime',
		name='Real-ESRGAN 4x (Anime)',
		description='AI upscaler optimized for anime/illustrations',
		suggested_denoise_strength=0.3,
		method=UpscalingMethod.AI,
	),
}


class ConfigService:
	def get_upscalers(self) -> list[UpscalerItem]:
		return [UPSCALER_METADATA[upscaler] for upscaler in UpscalerType]


config_service = ConfigService()
