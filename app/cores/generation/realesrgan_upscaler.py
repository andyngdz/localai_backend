"""Real-ESRGAN AI upscaler for high-quality image upscaling."""

from pathlib import Path

from basicsr.archs.rrdbnet_arch import RRDBNet
from PIL import Image
from pypdl import Pypdl
from realesrgan import RealESRGANer

from app.constants.upscalers import REALESRGAN_MODELS
from app.cores.generation.image_processor import image_processor
from app.cores.model_manager.resource_manager import resource_manager
from app.schemas.hires_fix import UpscalerType
from app.schemas.upscaler import UpscaleConfig
from app.services import device_service, logger_service, storage_service

logger = logger_service.get_logger(__name__, category='Upscaler')


class RealESRGANUpscaler:
	"""AI upscaler using Real-ESRGAN models. Loads per-request and unloads after."""

	def __init__(self) -> None:
		self._model = None

	def upscale(
		self,
		images: list[Image.Image],
		upscaler_type: UpscalerType,
		target_scale: float,
	) -> list[Image.Image]:
		"""Upscale images using Real-ESRGAN."""
		if not images:
			return []

		remote_model = REALESRGAN_MODELS[upscaler_type]
		original_width, original_height = images[0].size
		config = UpscaleConfig(
			batch_size=len(images),
			original_size=f'{original_width}x{original_height}',
			upscaler=upscaler_type.value,
			native_scale=remote_model.scale,
			target_scale=target_scale,
		)
		logger.info(f'AI upscaling\n{logger_service.format_config(config)}')

		try:
			self._load_model(upscaler_type)
			upscaled = self._upscale_images(images)
			upscaled = self._resize_to_target_scale(
				upscaled,
				original_width,
				original_height,
				target_scale,
				remote_model.scale,
			)
			return upscaled
		finally:
			self._cleanup()

	def _get_model_path(self, upscaler_type: UpscalerType) -> str:
		"""Download model if not cached, return local path."""
		remote_model = REALESRGAN_MODELS[upscaler_type]
		local_path = Path(storage_service.get_realesrgan_model_path(remote_model.filename))

		if local_path.exists():
			logger.info(f'Using cached model: {local_path}')
			return str(local_path)

		local_path.parent.mkdir(parents=True, exist_ok=True)

		logger.info(f'Downloading model from {remote_model.url}')

		downloader = Pypdl()
		downloader.start(remote_model.url, file_path=str(local_path), retries=3, etag_validation=True)

		logger.info(f'Model downloaded to: {local_path}')

		return str(local_path)

	def _load_model(self, upscaler_type: UpscalerType) -> None:
		"""Load Real-ESRGAN model for the specified type."""
		remote_model = REALESRGAN_MODELS[upscaler_type]
		model_path = self._get_model_path(upscaler_type)
		device = device_service.torch_device
		half_precision = device_service.is_cuda

		logger.info(f'Loading Real-ESRGAN model on {device} (half={half_precision})')

		network_model = self._create_network_model(upscaler_type, remote_model.scale)
		self._model = RealESRGANer(
			scale=remote_model.scale,
			model_path=model_path,
			model=network_model,
			tile=0,
			tile_pad=10,
			pre_pad=10,
			half=half_precision,
			device=device,
		)

	def _create_network_model(self, upscaler_type: UpscalerType, scale: int):
		"""Create the appropriate network architecture for the model type."""
		if upscaler_type == UpscalerType.REALESRGAN_X4PLUS_ANIME:
			return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=scale)

		return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)

	def _upscale_images(self, images: list[Image.Image]) -> list[Image.Image]:
		"""Upscale images using the loaded model."""
		if self._model is None:
			raise RuntimeError('Model not loaded')

		upscaled_images: list[Image.Image] = []

		for image in images:
			numpy_image = image_processor.pil_to_bgr_numpy(image)
			upscaled_numpy, _img_mode = self._model.enhance(numpy_image, outscale=self._model.scale)
			upscaled_pil = image_processor.bgr_numpy_to_pil(upscaled_numpy)
			upscaled_images.append(upscaled_pil)

		return upscaled_images

	def _resize_to_target_scale(
		self,
		images: list[Image.Image],
		original_width: int,
		original_height: int,
		target_scale: float,
		native_scale: int,
	) -> list[Image.Image]:
		"""Resize images to match target scale if different from native scale."""
		if target_scale == native_scale:
			return images

		target_width = int(original_width * target_scale)
		target_height = int(original_height * target_scale)
		logger.info(f'Resizing from native {native_scale}x to target {target_scale}x ({target_width}x{target_height})')

		resized = [img.resize((target_width, target_height), Image.Resampling.LANCZOS) for img in images]

		return resized

	def _cleanup(self) -> None:
		"""Clean up model and free GPU memory."""
		resource_manager.cleanup_pipeline(self._model, 'Real-ESRGAN')
		self._model = None


realesrgan_upscaler = RealESRGANUpscaler()
