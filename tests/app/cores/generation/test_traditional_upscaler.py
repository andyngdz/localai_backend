"""Tests for traditional upscaler."""

import pytest
from PIL import Image

from app.cores.generation.traditional_upscaler import TraditionalUpscaler
from app.schemas.hires_fix import UpscalerType


class TestPilUpscaling:
	"""Test PIL upscaling functionality."""

	@pytest.fixture
	def upscaler(self):
		"""Create upscaler instance."""
		return TraditionalUpscaler()

	@pytest.fixture
	def sample_images(self):
		"""Create sample PIL images [512x512]."""
		return [Image.new('RGB', (512, 512), color='red')]

	def test_upscale_2x_lanczos(self, upscaler, sample_images):
		"""Test 2x upscaling with Lanczos method."""
		result = upscaler._upscale_pil(sample_images, scale_factor=2.0, upscaler_type=UpscalerType.LANCZOS)

		assert len(result) == 1
		assert result[0].size == (1024, 1024)
		assert result[0].mode == 'RGB'

	def test_upscale_1_5x_bicubic(self, upscaler, sample_images):
		"""Test 1.5x upscaling with Bicubic method."""
		result = upscaler._upscale_pil(sample_images, scale_factor=1.5, upscaler_type=UpscalerType.BICUBIC)

		assert len(result) == 1
		assert result[0].size == (768, 768)

	def test_upscale_bilinear(self, upscaler, sample_images):
		"""Test upscaling with Bilinear method."""
		result = upscaler._upscale_pil(sample_images, scale_factor=2.0, upscaler_type=UpscalerType.BILINEAR)

		assert len(result) == 1
		assert result[0].size == (1024, 1024)

	def test_upscale_nearest(self, upscaler, sample_images):
		"""Test upscaling with Nearest method."""
		result = upscaler._upscale_pil(sample_images, scale_factor=2.0, upscaler_type=UpscalerType.NEAREST)

		assert len(result) == 1
		assert result[0].size == (1024, 1024)

	def test_upscale_batch(self, upscaler):
		"""Test upscaling with batch size > 1."""
		images = [Image.new('RGB', (512, 512), color='red') for _ in range(3)]
		result = upscaler._upscale_pil(images, scale_factor=2.0, upscaler_type=UpscalerType.LANCZOS)

		assert len(result) == 3
		for img in result:
			assert img.size == (1024, 1024)

	def test_different_aspect_ratios(self, upscaler):
		"""Test upscaling with non-square images."""
		images = [Image.new('RGB', (512, 768), color='blue')]
		result = upscaler._upscale_pil(images, scale_factor=2.0, upscaler_type=UpscalerType.LANCZOS)

		assert len(result) == 1
		assert result[0].size == (1024, 1536)


class TestUpscaleStepResolution:
	"""Test that upscale resolves hires_steps correctly."""

	@pytest.fixture
	def upscaler(self):
		"""Create upscaler instance."""
		return TraditionalUpscaler()

	@pytest.fixture
	def sample_images(self):
		"""Create sample PIL images."""
		return [Image.new('RGB', (512, 512), color='red')]

	def test_uses_hires_steps_when_nonzero(self, upscaler, sample_images):
		"""Test that hires_steps > 0 is used directly."""
		from unittest.mock import MagicMock, patch

		import torch

		from app.schemas.generators import GeneratorConfig
		from app.schemas.hires_fix import HiresFixConfig

		config = GeneratorConfig(
			prompt='test',
			width=512,
			height=512,
			steps=20,
			hires_fix=HiresFixConfig(
				upscale_factor=2.0,
				upscaler=UpscalerType.LANCZOS,
				denoising_strength=0.7,
				steps=15,
			),
		)
		mock_pipe = MagicMock()
		mock_output = MagicMock()
		mock_output.images = [Image.new('RGB', (1024, 1024))]
		mock_pipe.return_value = mock_output
		generator = torch.Generator().manual_seed(42)

		with patch.object(upscaler, 'refine', wraps=upscaler.refine) as mock_refine:
			mock_refine.return_value = [Image.new('RGB', (1024, 1024))]

			upscaler.upscale(
				config,
				mock_pipe,
				sample_images,
				generator,
				scale_factor=2.0,
				upscaler_type=UpscalerType.LANCZOS,
				hires_steps=15,
				denoising_strength=0.7,
			)

			call_args = mock_refine.call_args[0]
			assert call_args[4] == 15

	def test_uses_base_steps_when_hires_steps_zero(self, upscaler, sample_images):
		"""Test that hires_steps=0 falls back to config.steps."""
		from unittest.mock import MagicMock, patch

		import torch

		from app.schemas.generators import GeneratorConfig
		from app.schemas.hires_fix import HiresFixConfig

		config = GeneratorConfig(
			prompt='test',
			width=512,
			height=512,
			steps=20,
			hires_fix=HiresFixConfig(
				upscale_factor=2.0,
				upscaler=UpscalerType.LANCZOS,
				denoising_strength=0.7,
				steps=0,
			),
		)
		mock_pipe = MagicMock()
		mock_output = MagicMock()
		mock_output.images = [Image.new('RGB', (1024, 1024))]
		mock_pipe.return_value = mock_output
		generator = torch.Generator().manual_seed(42)

		with patch.object(upscaler, 'refine', wraps=upscaler.refine) as mock_refine:
			mock_refine.return_value = [Image.new('RGB', (1024, 1024))]

			upscaler.upscale(
				config,
				mock_pipe,
				sample_images,
				generator,
				scale_factor=2.0,
				upscaler_type=UpscalerType.LANCZOS,
				hires_steps=0,
				denoising_strength=0.7,
			)

			call_args = mock_refine.call_args[0]
			assert call_args[4] == 20
