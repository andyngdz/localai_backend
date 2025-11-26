"""Tests for latent upscaler."""

import pytest
import torch

from app.cores.generation.upscaler import LatentUpscaler
from app.schemas.hires_fix import UpscalerType


class TestLatentUpscaler:
	"""Test latent upscaling functionality."""

	@pytest.fixture
	def upscaler(self):
		"""Create upscaler instance."""
		return LatentUpscaler()

	@pytest.fixture
	def sample_latents(self):
		"""Create sample latent tensor [batch=1, channels=4, h=64, w=64]."""
		return torch.randn(1, 4, 64, 64)

	def test_upscale_2x_latent(self, upscaler, sample_latents):
		"""Test 2x upscaling with Latent (bilinear) method."""
		result = upscaler.upscale(sample_latents, scale_factor=2.0, upscaler_type=UpscalerType.LATENT)

		assert result.shape == (1, 4, 128, 128)
		assert result.dtype == sample_latents.dtype
		assert result.device == sample_latents.device

	def test_upscale_1_5x_latent(self, upscaler, sample_latents):
		"""Test 1.5x upscaling with Latent (bilinear) method."""
		result = upscaler.upscale(sample_latents, scale_factor=1.5, upscaler_type=UpscalerType.LATENT)

		assert result.shape == (1, 4, 96, 96)

	def test_upscale_latent_nearest(self, upscaler, sample_latents):
		"""Test upscaling with Latent (nearest) method."""
		result = upscaler.upscale(sample_latents, scale_factor=2.0, upscaler_type=UpscalerType.LATENT_NEAREST)

		assert result.shape == (1, 4, 128, 128)

	def test_upscale_latent_nearest_exact(self, upscaler, sample_latents):
		"""Test upscaling with Latent (nearest-exact) method."""
		result = upscaler.upscale(sample_latents, scale_factor=2.0, upscaler_type=UpscalerType.LATENT_NEAREST_EXACT)

		assert result.shape == (1, 4, 128, 128)

	def test_upscale_batch(self, upscaler):
		"""Test upscaling with batch size > 1."""
		latents = torch.randn(3, 4, 64, 64)
		result = upscaler.upscale(latents, scale_factor=2.0)

		assert result.shape == (3, 4, 128, 128)

	def test_scale_factor_validation(self, upscaler, sample_latents):
		"""Test that scale_factor <= 1.0 raises error."""
		with pytest.raises(ValueError, match='scale_factor must be > 1.0'):
			upscaler.upscale(sample_latents, scale_factor=1.0)

		with pytest.raises(ValueError, match='scale_factor must be > 1.0'):
			upscaler.upscale(sample_latents, scale_factor=0.5)

	def test_preserves_gradient(self, upscaler):
		"""Test that upscaling preserves gradient flow."""
		latents = torch.randn(1, 4, 64, 64, requires_grad=True)
		result = upscaler.upscale(latents, scale_factor=2.0)

		assert result.requires_grad
		loss = result.sum()
		loss.backward()
		assert latents.grad is not None

	def test_different_aspect_ratios(self, upscaler):
		"""Test upscaling with non-square latents."""
		latents = torch.randn(1, 4, 64, 96)
		result = upscaler.upscale(latents, scale_factor=2.0)

		assert result.shape == (1, 4, 128, 192)

	def test_default_upscaler_type(self, upscaler, sample_latents):
		"""Test that default upscaler type is LATENT (bilinear)."""
		result = upscaler.upscale(sample_latents, scale_factor=2.0)

		assert result.shape == (1, 4, 128, 128)
