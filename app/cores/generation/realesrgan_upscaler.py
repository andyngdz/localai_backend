"""Real-ESRGAN AI upscaler for high-quality image upscaling."""

import gc
import sys
import types
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torchvision.transforms.functional import rgb_to_grayscale

from app.constants.upscalers import REALESRGAN_MODELS
from app.schemas.hires_fix import UpscalerType
from app.services import device_service, logger_service
from config import CACHE_FOLDER

# Patch basicsr's deprecated torchvision import for newer torchvision versions.
# basicsr uses `torchvision.transforms.functional_tensor.rgb_to_grayscale` which was
# removed in torchvision 0.18+. Create a shim module with the expected function.
_functional_tensor = types.ModuleType('torchvision.transforms.functional_tensor')
setattr(_functional_tensor, 'rgb_to_grayscale', rgb_to_grayscale)
sys.modules['torchvision.transforms.functional_tensor'] = _functional_tensor

logger = logger_service.get_logger(__name__, category='Upscaler')


class RealESRGANUpscaler:
	"""AI upscaler using Real-ESRGAN models.

	Loads models per-request and unloads after to minimize VRAM usage.
	Supports CUDA, MPS, and CPU devices.
	"""

	def __init__(self) -> None:
		self._model = None

	def upscale(
		self,
		images: list[Image.Image],
		upscaler_type: UpscalerType,
		target_scale: float,
	) -> list[Image.Image]:
		"""Upscale images using Real-ESRGAN.

		Args:
			images: List of PIL images to upscale
			upscaler_type: Real-ESRGAN model variant to use
			target_scale: Desired scale factor (may differ from model's native scale)

		Returns:
			List of upscaled PIL images
		"""
		if not images:
			return []

		remote_model = REALESRGAN_MODELS[upscaler_type]
		original_width, original_height = images[0].size
		logger.info(
			f'AI upscaling {len(images)} image(s) from {original_width}x{original_height} '
			f'using {upscaler_type.value} (native {remote_model.scale}x, target {target_scale}x)'
		)

		try:
			self._load_model(upscaler_type)
			upscaled = self._upscale_images(images)
			upscaled = self._resize_to_target_scale(
				upscaled, original_width, original_height, target_scale, remote_model.scale
			)
			return upscaled
		finally:
			self._cleanup()

	def _get_model_path(self, upscaler_type: UpscalerType) -> str:
		"""Download model if not cached, return local path."""
		from pypdl import Pypdl

		remote_model = REALESRGAN_MODELS[upscaler_type]
		cache_dir = Path(CACHE_FOLDER) / 'realesrgan'
		local_path = cache_dir / remote_model.filename

		if local_path.exists():
			logger.info(f'Using cached model: {local_path}')
			return str(local_path)

		cache_dir.mkdir(parents=True, exist_ok=True)
		logger.info(f'Downloading model from {remote_model.url}')
		downloader = Pypdl()
		downloader.start(remote_model.url, file_path=str(local_path), retries=3, etag_validation=True)
		logger.info(f'Model downloaded to: {local_path}')
		return str(local_path)

	def _load_model(self, upscaler_type: UpscalerType) -> None:
		"""Load Real-ESRGAN model for the specified type."""
		from realesrgan import RealESRGANer

		remote_model = REALESRGAN_MODELS[upscaler_type]
		model_path = self._get_model_path(upscaler_type)

		device = self._get_device()
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
		from basicsr.archs.rrdbnet_arch import RRDBNet

		if upscaler_type == UpscalerType.REALESRGAN_X4PLUS_ANIME:
			return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=scale)
		return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)

	def _get_device(self) -> torch.device:
		"""Get the appropriate torch device."""
		if device_service.is_cuda:
			return torch.device('cuda')
		if device_service.is_mps:
			return torch.device('mps')
		return torch.device('cpu')

	def _upscale_images(self, images: list[Image.Image]) -> list[Image.Image]:
		"""Upscale images using the loaded model."""
		if self._model is None:
			raise RuntimeError('Model not loaded')

		upscaled_images: list[Image.Image] = []
		for image in images:
			numpy_image = self._pil_to_numpy(image)
			upscaled_numpy, _ = self._model.enhance(numpy_image, outscale=self._model.scale)
			upscaled_pil = self._numpy_to_pil(upscaled_numpy)
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

	def _pil_to_numpy(self, image: Image.Image) -> NDArray[np.uint8]:
		"""Convert PIL image to numpy array in BGR format for Real-ESRGAN."""
		rgb_array = np.array(image)
		bgr_array: NDArray[np.uint8] = rgb_array[:, :, ::-1]
		return bgr_array

	def _numpy_to_pil(self, array: NDArray[np.uint8]) -> Image.Image:
		"""Convert numpy array in BGR format back to PIL image."""
		rgb_array = array[:, :, ::-1]
		return Image.fromarray(rgb_array)

	def _cleanup(self) -> None:
		"""Clean up model and free GPU memory."""
		logger.info('Cleaning up Real-ESRGAN model')
		if self._model is not None:
			del self._model
			self._model = None

		gc.collect()

		if device_service.is_cuda:
			torch.cuda.synchronize()
			torch.cuda.empty_cache()
		elif device_service.is_mps:
			torch.mps.synchronize()
			torch.mps.empty_cache()

		gc.collect()
		logger.info('Real-ESRGAN cleanup completed')


realesrgan_upscaler = RealESRGANUpscaler()
