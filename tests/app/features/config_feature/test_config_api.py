"""Tests for app/features/config/api.py

Covers:
- GET /config endpoint returns ConfigResponse
- Response includes upscalers sections grouped by method
- Safety check toggle endpoint
"""

from unittest.mock import MagicMock, patch

from app.features.config.api import get_config, update_safety_check
from app.schemas.config import (
	ConfigResponse,
	SafetyCheckRequest,
	UpscalerItem,
	UpscalerSection,
	UpscalingMethod,
)


class TestConfigAPI:
	"""Test config API endpoints."""

	@patch('app.features.config.api.config_crud')
	def test_get_config_returns_config_response(self, mock_config_crud):
		"""Test that get_config returns a ConfigResponse."""
		mock_config_crud.get_safety_check_enabled.return_value = True
		mock_db = MagicMock()

		result = get_config(mock_db)

		assert isinstance(result, ConfigResponse)

	@patch('app.features.config.api.config_crud')
	def test_get_config_includes_upscalers_sections(self, mock_config_crud):
		"""Test that get_config response includes upscalers sections."""
		mock_config_crud.get_safety_check_enabled.return_value = True
		mock_db = MagicMock()

		result = get_config(mock_db)

		assert hasattr(result, 'upscalers')
		assert isinstance(result.upscalers, list)
		assert len(result.upscalers) > 0

	@patch('app.features.config.api.config_crud')
	def test_get_config_upscalers_are_upscaler_sections(self, mock_config_crud):
		"""Test that upscalers in response are UpscalerSection instances."""
		mock_config_crud.get_safety_check_enabled.return_value = True
		mock_db = MagicMock()

		result = get_config(mock_db)

		assert all(isinstance(section, UpscalerSection) for section in result.upscalers)

	@patch('app.features.config.api.config_crud')
	def test_get_config_sections_have_options(self, mock_config_crud):
		"""Test that each section has options that are UpscalerItem instances."""
		mock_config_crud.get_safety_check_enabled.return_value = True
		mock_db = MagicMock()

		result = get_config(mock_db)

		for section in result.upscalers:
			assert hasattr(section, 'options')
			assert isinstance(section.options, list)
			assert all(isinstance(item, UpscalerItem) for item in section.options)

	@patch('app.features.config.api.config_crud')
	@patch('app.features.config.api.config_service')
	def test_get_config_calls_service(self, mock_config_service, mock_config_crud):
		"""Test that get_config calls config_service.get_upscaler_sections."""
		mock_config_crud.get_safety_check_enabled.return_value = True
		mock_db = MagicMock()
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

		result = get_config(mock_db)

		mock_config_service.get_upscaler_sections.assert_called_once()
		assert result.upscalers == mock_sections


class TestSafetyCheckAPI:
	"""Test safety check toggle API endpoints."""

	@patch('app.features.config.api.config_crud')
	def test_get_config_includes_safety_check_enabled(self, mock_config_crud):
		"""Test that get_config returns safety_check_enabled field."""
		mock_config_crud.get_safety_check_enabled.return_value = True
		mock_db = MagicMock()

		result = get_config(mock_db)

		assert hasattr(result, 'safety_check_enabled')
		assert result.safety_check_enabled is True

	@patch('app.features.config.api.config_crud')
	def test_get_config_returns_safety_check_disabled(self, mock_config_crud):
		"""Test that get_config returns safety_check_enabled=False when disabled."""
		mock_config_crud.get_safety_check_enabled.return_value = False
		mock_db = MagicMock()

		result = get_config(mock_db)

		assert result.safety_check_enabled is False

	@patch('app.features.config.api.config_crud')
	def test_update_safety_check_enables(self, mock_config_crud):
		"""Test enabling safety check via PUT endpoint."""
		mock_config_crud.get_safety_check_enabled.return_value = True
		mock_db = MagicMock()
		request = SafetyCheckRequest(enabled=True)

		result = update_safety_check(request, mock_db)

		mock_config_crud.set_safety_check_enabled.assert_called_once_with(mock_db, True)
		assert result.safety_check_enabled is True

	@patch('app.features.config.api.config_crud')
	def test_update_safety_check_disables(self, mock_config_crud):
		"""Test disabling safety check via PUT endpoint."""
		mock_config_crud.get_safety_check_enabled.return_value = False
		mock_db = MagicMock()
		request = SafetyCheckRequest(enabled=False)

		result = update_safety_check(request, mock_db)

		mock_config_crud.set_safety_check_enabled.assert_called_once_with(mock_db, False)
		assert result.safety_check_enabled is False
