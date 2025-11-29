"""Tests for Real-ESRGAN AI upscaler."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from app.schemas.hires_fix import UpscalerType


class TestRealESRGANUpscaler:
	"""Test Real-ESRGAN upscaling functionality."""

	@pytest.fixture
	def upscaler(self):
		"""Create upscaler instance."""
		from app.cores.generation.realesrgan_upscaler import RealESRGANUpscaler

		return RealESRGANUpscaler()

	@pytest.fixture
	def sample_images(self):
		"""Create sample PIL images [512x512]."""
		return [Image.new('RGB', (512, 512), color='red')]

	@pytest.fixture
	def mock_realesrganer(self):
		"""Create mock RealESRGANer that returns upscaled numpy array."""
		mock = MagicMock()
		upscaled_array = np.zeros((1024, 1024, 3), dtype=np.uint8)
		mock.enhance.return_value = (upscaled_array, None)
		mock.scale = 2
		return mock

	@pytest.fixture
	def mock_cached_model_path(self, tmp_path):
		"""Create a mock cached model path."""
		model_path = tmp_path / 'realesrgan' / 'RealESRGAN_x2plus.pth'
		model_path.parent.mkdir(parents=True, exist_ok=True)
		model_path.touch()
		return tmp_path

	def test_upscale_empty_list(self, upscaler):
		"""Test that empty image list returns empty list."""
		result = upscaler.upscale([], UpscalerType.REALESRGAN_X2PLUS, 2.0)
		assert result == []

	def test_upscale_x2plus_model(self, upscaler, sample_images, tmp_path):
		"""Test upscaling with x2plus model."""
		model_path = tmp_path / 'realesrgan' / 'RealESRGAN_x2plus.pth'
		model_path.parent.mkdir(parents=True, exist_ok=True)
		model_path.touch()

		mock_rrdbnet = MagicMock()
		mock_realesrganer_class = MagicMock()

		upscaled_array = np.zeros((1024, 1024, 3), dtype=np.uint8)
		mock_model = MagicMock()
		mock_model.enhance.return_value = (upscaled_array, None)
		mock_model.scale = 2
		mock_realesrganer_class.return_value = mock_model

		with (
			patch.dict(
				sys.modules,
				{
					'basicsr': MagicMock(),
					'basicsr.archs': MagicMock(),
					'basicsr.archs.rrdbnet_arch': MagicMock(RRDBNet=mock_rrdbnet),
				},
			),
			patch.dict(sys.modules, {'realesrgan': MagicMock(RealESRGANer=mock_realesrganer_class)}),
			patch('config.CACHE_FOLDER', str(tmp_path)),
		):
			result = upscaler.upscale(sample_images, UpscalerType.REALESRGAN_X2PLUS, 2.0)

			assert len(result) == 1
			assert isinstance(result[0], Image.Image)

	def test_upscale_x4plus_model(self, upscaler, sample_images, tmp_path):
		"""Test upscaling with x4plus model."""
		model_path = tmp_path / 'realesrgan' / 'RealESRGAN_x4plus.pth'
		model_path.parent.mkdir(parents=True, exist_ok=True)
		model_path.touch()

		mock_rrdbnet = MagicMock()
		mock_realesrganer_class = MagicMock()

		upscaled_array = np.zeros((2048, 2048, 3), dtype=np.uint8)
		mock_model = MagicMock()
		mock_model.enhance.return_value = (upscaled_array, None)
		mock_model.scale = 4
		mock_realesrganer_class.return_value = mock_model

		with (
			patch.dict(
				sys.modules,
				{
					'basicsr': MagicMock(),
					'basicsr.archs': MagicMock(),
					'basicsr.archs.rrdbnet_arch': MagicMock(RRDBNet=mock_rrdbnet),
				},
			),
			patch.dict(sys.modules, {'realesrgan': MagicMock(RealESRGANer=mock_realesrganer_class)}),
			patch('config.CACHE_FOLDER', str(tmp_path)),
		):
			result = upscaler.upscale(sample_images, UpscalerType.REALESRGAN_X4PLUS, 4.0)

			assert len(result) == 1
			assert isinstance(result[0], Image.Image)

	def test_upscale_anime_model(self, upscaler, sample_images, tmp_path):
		"""Test upscaling with anime model."""
		model_path = tmp_path / 'realesrgan' / 'RealESRGAN_x4plus_anime_6B.pth'
		model_path.parent.mkdir(parents=True, exist_ok=True)
		model_path.touch()

		mock_rrdbnet = MagicMock()
		mock_realesrganer_class = MagicMock()

		upscaled_array = np.zeros((2048, 2048, 3), dtype=np.uint8)
		mock_model = MagicMock()
		mock_model.enhance.return_value = (upscaled_array, None)
		mock_model.scale = 4
		mock_realesrganer_class.return_value = mock_model

		with (
			patch.dict(
				sys.modules,
				{
					'basicsr': MagicMock(),
					'basicsr.archs': MagicMock(),
					'basicsr.archs.rrdbnet_arch': MagicMock(RRDBNet=mock_rrdbnet),
				},
			),
			patch.dict(sys.modules, {'realesrgan': MagicMock(RealESRGANer=mock_realesrganer_class)}),
			patch('config.CACHE_FOLDER', str(tmp_path)),
		):
			result = upscaler.upscale(sample_images, UpscalerType.REALESRGAN_X4PLUS_ANIME, 4.0)

			assert len(result) == 1
			assert isinstance(result[0], Image.Image)
			mock_rrdbnet.assert_called_with(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)

	def test_cleanup_called_after_upscale(self, upscaler, sample_images, tmp_path):
		"""Test that cleanup is called after upscaling."""
		model_path = tmp_path / 'realesrgan' / 'RealESRGAN_x2plus.pth'
		model_path.parent.mkdir(parents=True, exist_ok=True)
		model_path.touch()

		mock_rrdbnet = MagicMock()
		mock_realesrganer_class = MagicMock()

		upscaled_array = np.zeros((1024, 1024, 3), dtype=np.uint8)
		mock_model = MagicMock()
		mock_model.enhance.return_value = (upscaled_array, None)
		mock_model.scale = 2
		mock_realesrganer_class.return_value = mock_model

		with (
			patch.dict(
				sys.modules,
				{
					'basicsr': MagicMock(),
					'basicsr.archs': MagicMock(),
					'basicsr.archs.rrdbnet_arch': MagicMock(RRDBNet=mock_rrdbnet),
				},
			),
			patch.dict(sys.modules, {'realesrgan': MagicMock(RealESRGANer=mock_realesrganer_class)}),
			patch('config.CACHE_FOLDER', str(tmp_path)),
		):
			upscaler.upscale(sample_images, UpscalerType.REALESRGAN_X2PLUS, 2.0)

			assert upscaler._model is None

	def test_cleanup_called_on_error(self, upscaler, sample_images, tmp_path):
		"""Test that cleanup is called even when upscaling fails."""
		model_path = tmp_path / 'realesrgan' / 'RealESRGAN_x2plus.pth'
		model_path.parent.mkdir(parents=True, exist_ok=True)
		model_path.touch()

		mock_rrdbnet = MagicMock()
		mock_realesrganer_class = MagicMock()

		mock_model = MagicMock()
		mock_model.enhance.side_effect = RuntimeError('Upscaling failed')
		mock_model.scale = 2
		mock_realesrganer_class.return_value = mock_model

		with (
			patch.dict(
				sys.modules,
				{
					'basicsr': MagicMock(),
					'basicsr.archs': MagicMock(),
					'basicsr.archs.rrdbnet_arch': MagicMock(RRDBNet=mock_rrdbnet),
				},
			),
			patch.dict(sys.modules, {'realesrgan': MagicMock(RealESRGANer=mock_realesrganer_class)}),
			patch('config.CACHE_FOLDER', str(tmp_path)),
		):
			with pytest.raises(RuntimeError):
				upscaler.upscale(sample_images, UpscalerType.REALESRGAN_X2PLUS, 2.0)

			assert upscaler._model is None

	def test_target_scale_resize(self, upscaler, sample_images, tmp_path):
		"""Test that images are resized when target scale differs from native."""
		model_path = tmp_path / 'realesrgan' / 'RealESRGAN_x4plus.pth'
		model_path.parent.mkdir(parents=True, exist_ok=True)
		model_path.touch()

		mock_rrdbnet = MagicMock()
		mock_realesrganer_class = MagicMock()

		upscaled_array = np.zeros((2048, 2048, 3), dtype=np.uint8)
		mock_model = MagicMock()
		mock_model.enhance.return_value = (upscaled_array, None)
		mock_model.scale = 4
		mock_realesrganer_class.return_value = mock_model

		with (
			patch.dict(
				sys.modules,
				{
					'basicsr': MagicMock(),
					'basicsr.archs': MagicMock(),
					'basicsr.archs.rrdbnet_arch': MagicMock(RRDBNet=mock_rrdbnet),
				},
			),
			patch.dict(sys.modules, {'realesrgan': MagicMock(RealESRGANer=mock_realesrganer_class)}),
			patch('config.CACHE_FOLDER', str(tmp_path)),
		):
			result = upscaler.upscale(sample_images, UpscalerType.REALESRGAN_X4PLUS, 3.0)

			assert len(result) == 1
			assert result[0].size == (1536, 1536)

	def test_model_download_when_not_cached(self, upscaler, sample_images, tmp_path):
		"""Test that model is downloaded when not cached."""
		mock_downloader = MagicMock()
		mock_pypdl_class = MagicMock(return_value=mock_downloader)
		mock_rrdbnet = MagicMock()
		mock_realesrganer_class = MagicMock()

		upscaled_array = np.zeros((1024, 1024, 3), dtype=np.uint8)
		mock_model = MagicMock()
		mock_model.enhance.return_value = (upscaled_array, None)
		mock_model.scale = 2
		mock_realesrganer_class.return_value = mock_model

		# Create mock pypdl module
		mock_pypdl_module = MagicMock()
		mock_pypdl_module.Pypdl = mock_pypdl_class

		with (
			patch.dict(sys.modules, {'pypdl': mock_pypdl_module}),
			patch.dict(
				sys.modules,
				{
					'basicsr': MagicMock(),
					'basicsr.archs': MagicMock(),
					'basicsr.archs.rrdbnet_arch': MagicMock(RRDBNet=mock_rrdbnet),
				},
			),
			patch.dict(sys.modules, {'realesrgan': MagicMock(RealESRGANer=mock_realesrganer_class)}),
			patch('config.CACHE_FOLDER', str(tmp_path)),
		):
			# Force re-import to pick up mocked module
			import importlib

			import app.cores.generation.realesrgan_upscaler

			importlib.reload(app.cores.generation.realesrgan_upscaler)
			fresh_upscaler = app.cores.generation.realesrgan_upscaler.RealESRGANUpscaler()

			fresh_upscaler.upscale(sample_images, UpscalerType.REALESRGAN_X2PLUS, 2.0)

			mock_downloader.start.assert_called_once()

	def test_pil_to_numpy_conversion(self, upscaler):
		"""Test PIL to numpy conversion produces BGR array."""
		pil_image = Image.new('RGB', (100, 100), color=(255, 0, 0))
		numpy_array = upscaler._pil_to_numpy(pil_image)

		assert numpy_array.shape == (100, 100, 3)
		assert numpy_array[0, 0, 0] == 0
		assert numpy_array[0, 0, 2] == 255

	def test_numpy_to_pil_conversion(self, upscaler):
		"""Test numpy to PIL conversion produces RGB image."""
		bgr_array = np.zeros((100, 100, 3), dtype=np.uint8)
		bgr_array[:, :, 0] = 255  # Blue channel in BGR
		pil_image = upscaler._numpy_to_pil(bgr_array)

		assert pil_image.size == (100, 100)
		assert pil_image.mode == 'RGB'
		# BGR [255, 0, 0] -> RGB [0, 0, 255] (blue)
		assert pil_image.getpixel((0, 0)) == (0, 0, 255)
