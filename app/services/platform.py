from app.constants.platform import OperatingSystem
from app.services.logger import logger_service
from app.services.patch_service import WindowsProactorEventLoopPolicy

logger = logger_service.get_logger(__name__, category='Service')


class PlatformService:
	"""Service to handle platform-specific operations."""

	def init(self):
		"""Initialize the platform service."""
		try:
			os_type = OperatingSystem.from_sys_platform()
		except ValueError:
			return

		if os_type == OperatingSystem.WINDOWS and WindowsProactorEventLoopPolicy is not None:
			try:
				from app.services.patch_service import setup_windows_event_loop

				setup_windows_event_loop()
				logger.info('Windows event loop policy set for compatibility.')
			except Exception as error:
				logger.warning(f'Failed to set event loop policy: {error}')


platform_service = PlatformService()
