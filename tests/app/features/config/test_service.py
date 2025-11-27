"""Tests for app/features/config/service.py

Covers:
- ConfigService.get_upscalers returns all upscaler types
- Each upscaler has correct metadata
"""

import pytest

from app.features.config.service import UPSCALER_METADATA, ConfigService
from app.schemas.config import UpscalerItem
from app.schemas.hires_fix import UpscalerType


class TestConfigService:
	"""Test ConfigService functionality."""

	def setup_method(self):
		"""Setup test method."""
		self.service = ConfigService()

	def test_get_upscalers_returns_all_upscaler_types(self):
		"""Test that get_upscalers returns all UpscalerType enum values."""
		result = self.service.get_upscalers()

		assert len(result) == len(UpscalerType)
		values = [item.value for item in result]
		for upscaler_type in UpscalerType:
			assert upscaler_type.value in values

	def test_get_upscalers_returns_upscaler_items(self):
		"""Test that get_upscalers returns UpscalerItem instances."""
		result = self.service.get_upscalers()

		assert all(isinstance(item, UpscalerItem) for item in result)

	def test_get_upscalers_includes_correct_metadata(self):
		"""Test that each upscaler has correct metadata from UPSCALER_METADATA."""
		result = self.service.get_upscalers()

		for item in result:
			upscaler_type = UpscalerType(item.value)
			expected = UPSCALER_METADATA[upscaler_type]

			assert item.name == expected.name
			assert item.description == expected.description
			assert item.suggested_denoise_strength == pytest.approx(expected.suggested_denoise_strength)

	def test_lanczos_upscaler_metadata(self):
		"""Test Lanczos upscaler has correct metadata."""
		result = self.service.get_upscalers()
		lanczos = next(item for item in result if item.value == 'Lanczos')

		assert lanczos.name == 'Lanczos (High Quality)'
		assert lanczos.description == 'High-quality resampling, best for photos'
		assert lanczos.suggested_denoise_strength == pytest.approx(0.4)

	def test_nearest_upscaler_metadata(self):
		"""Test Nearest upscaler has correct metadata."""
		result = self.service.get_upscalers()
		nearest = next(item for item in result if item.value == 'Nearest')

		assert nearest.name == 'Nearest (Sharp Edges)'
		assert nearest.description == 'No interpolation, preserves sharp edges'
		assert nearest.suggested_denoise_strength == pytest.approx(0.3)

	def test_denoise_strengths_are_in_valid_range(self):
		"""Test that all denoise strengths are between 0 and 1."""
		result = self.service.get_upscalers()

		for item in result:
			assert 0.0 <= item.suggested_denoise_strength <= 1.0
