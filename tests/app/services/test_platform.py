"""Tests for the platform service."""

from app.services.platform import PlatformService, platform_service


class TestPlatformService:
	"""Test the PlatformService class."""

	def test_platform_service_singleton(self):
		"""Test that platform_service is a singleton instance."""
		assert isinstance(platform_service, PlatformService)

	def test_init_runs_successfully(self):
		"""Test that init() runs without errors."""
		service = PlatformService()

		# Should not raise any exceptions
		try:
			service.init()
			success = True
		except Exception:
			success = False

		assert success

	def test_module_imports_successfully(self):
		"""Test that the platform module imports successfully."""
		# This test ensures the module-level logger initialization is executed
		from app.services import platform

		assert platform is not None
		assert hasattr(platform, 'platform_service')
