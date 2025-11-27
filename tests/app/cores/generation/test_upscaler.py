"""Tests for image upscaler."""

import pytest
from PIL import Image

from app.cores.generation.upscaler import ImageUpscaler
from app.schemas.hires_fix import UpscalerType


class TestImageUpscaler:
	"""Test image upscaling functionality."""

	@pytest.fixture
	def upscaler(self):
		"""Create upscaler instance."""
		return ImageUpscaler()

	@pytest.fixture
	def sample_images(self):
		"""Create sample PIL images [512x512]."""
		return [Image.new('RGB', (512, 512), color='red')]

	def test_upscale_2x_lanczos(self, upscaler, sample_images):
		"""Test 2x upscaling with Lanczos method."""
		result = upscaler.upscale(sample_images, scale_factor=2.0, upscaler_type=UpscalerType.LANCZOS)

		assert len(result) == 1
		assert result[0].size == (1024, 1024)
		assert result[0].mode == 'RGB'

	def test_upscale_1_5x_bicubic(self, upscaler, sample_images):
		"""Test 1.5x upscaling with Bicubic method."""
		result = upscaler.upscale(sample_images, scale_factor=1.5, upscaler_type=UpscalerType.BICUBIC)

		assert len(result) == 1
		assert result[0].size == (768, 768)

	def test_upscale_bilinear(self, upscaler, sample_images):
		"""Test upscaling with Bilinear method."""
		result = upscaler.upscale(sample_images, scale_factor=2.0, upscaler_type=UpscalerType.BILINEAR)

		assert len(result) == 1
		assert result[0].size == (1024, 1024)

	def test_upscale_nearest(self, upscaler, sample_images):
		"""Test upscaling with Nearest method."""
		result = upscaler.upscale(sample_images, scale_factor=2.0, upscaler_type=UpscalerType.NEAREST)

		assert len(result) == 1
		assert result[0].size == (1024, 1024)

	def test_upscale_batch(self, upscaler):
		"""Test upscaling with batch size > 1."""
		images = [Image.new('RGB', (512, 512), color='red') for _ in range(3)]
		result = upscaler.upscale(images, scale_factor=2.0)

		assert len(result) == 3
		for img in result:
			assert img.size == (1024, 1024)

	def test_scale_factor_validation(self, upscaler, sample_images):
		"""Test that scale_factor <= 1.0 raises error."""
		with pytest.raises(ValueError, match='scale_factor must be > 1.0'):
			upscaler.upscale(sample_images, scale_factor=1.0)

		with pytest.raises(ValueError, match='scale_factor must be > 1.0'):
			upscaler.upscale(sample_images, scale_factor=0.5)

	def test_different_aspect_ratios(self, upscaler):
		"""Test upscaling with non-square images."""
		images = [Image.new('RGB', (512, 768), color='blue')]
		result = upscaler.upscale(images, scale_factor=2.0)

		assert len(result) == 1
		assert result[0].size == (1024, 1536)

	def test_default_upscaler_type(self, upscaler, sample_images):
		"""Test that default upscaler type is LANCZOS."""
		result = upscaler.upscale(sample_images, scale_factor=2.0)

		assert len(result) == 1
		assert result[0].size == (1024, 1024)

	def test_empty_image_list(self, upscaler):
		"""Test that empty image list returns empty list."""
		result = upscaler.upscale([], scale_factor=2.0)

		assert result == []
