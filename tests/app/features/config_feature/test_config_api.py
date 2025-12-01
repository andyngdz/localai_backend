"""Tests for app/features/config/api.py

Covers:
- GET /config endpoint returns ConfigResponse
- Response includes upscalers list
"""

from unittest.mock import patch

from app.features.config.api import get_config
from app.schemas.config import ConfigResponse, UpscalerItem


class TestConfigAPI:
	"""Test config API endpoints."""

	def test_get_config_returns_config_response(self):
		"""Test that get_config returns a ConfigResponse."""
		result = get_config()

		assert isinstance(result, ConfigResponse)

	def test_get_config_includes_upscalers(self):
		"""Test that get_config response includes upscalers."""
		result = get_config()

		assert hasattr(result, 'upscalers')
		assert isinstance(result.upscalers, list)
		assert len(result.upscalers) > 0

	def test_get_config_upscalers_are_upscaler_items(self):
		"""Test that upscalers in response are UpscalerItem instances."""
		result = get_config()

		assert all(isinstance(item, UpscalerItem) for item in result.upscalers)

	@patch('app.features.config.api.config_service')
	def test_get_config_calls_service(self, mock_config_service):
		"""Test that get_config calls config_service.get_upscalers."""
		from app.schemas.config import UpscalingMethod

		mock_upscalers = [
			UpscalerItem(
				value='Test',
				name='Test Upscaler',
				description='Test description',
				suggested_denoise_strength=0.5,
				method=UpscalingMethod.TRADITIONAL,
				is_recommended=False,
			)
		]
		mock_config_service.get_upscalers.return_value = mock_upscalers

		result = get_config()

		mock_config_service.get_upscalers.assert_called_once()
		assert result.upscalers == mock_upscalers
