"""Tests for app/features/config/api.py

Covers:
- GET /config endpoint returns ConfigResponse
- Response includes upscalers sections grouped by method
"""

from unittest.mock import patch

from app.features.config.api import get_config
from app.schemas.config import ConfigResponse, UpscalerItem, UpscalerSection, UpscalingMethod


class TestConfigAPI:
	"""Test config API endpoints."""

	def test_get_config_returns_config_response(self):
		"""Test that get_config returns a ConfigResponse."""
		result = get_config()

		assert isinstance(result, ConfigResponse)

	def test_get_config_includes_upscalers_sections(self):
		"""Test that get_config response includes upscalers sections."""
		result = get_config()

		assert hasattr(result, 'upscalers')
		assert isinstance(result.upscalers, list)
		assert len(result.upscalers) > 0

	def test_get_config_upscalers_are_upscaler_sections(self):
		"""Test that upscalers in response are UpscalerSection instances."""
		result = get_config()

		assert all(isinstance(section, UpscalerSection) for section in result.upscalers)

	def test_get_config_sections_have_options(self):
		"""Test that each section has options that are UpscalerItem instances."""
		result = get_config()

		for section in result.upscalers:
			assert hasattr(section, 'options')
			assert isinstance(section.options, list)
			assert all(isinstance(item, UpscalerItem) for item in section.options)

	@patch('app.features.config.api.config_service')
	def test_get_config_calls_service(self, mock_config_service):
		"""Test that get_config calls config_service.get_upscaler_sections."""
		mock_sections = [
			UpscalerSection(
				method=UpscalingMethod.TRADITIONAL,
				title='Traditional',
				options=[
					UpscalerItem(
						value='Test',
						name='Test Upscaler',
						description='Test description',
						suggested_denoise_strength=0.5,
						method=UpscalingMethod.TRADITIONAL,
						is_recommended=False,
					)
				],
			)
		]
		mock_config_service.get_upscaler_sections.return_value = mock_sections

		result = get_config()

		mock_config_service.get_upscaler_sections.assert_called_once()
		assert result.upscalers == mock_sections
