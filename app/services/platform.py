import asyncio
import sys

from app.constants.platform import OperatingSystem
from app.services.logger import logger_service

logger = logger_service.get_logger(__name__, category='Service')

# Windows-specific event loop policy - only available on Windows
# Using sys.platform here for Pyright type narrowing at module level
if sys.platform == 'win32':
	WindowsProactorEventLoopPolicy = asyncio.WindowsProactorEventLoopPolicy
else:
	WindowsProactorEventLoopPolicy = None


class PlatformService:
	"""Service to handle platform-specific operations."""

	def init(self):
		"""Initialize the platform service."""

		try:
			os_type = OperatingSystem.from_sys_platform()
		except ValueError:
			# If platform detection fails, skip event loop policy setup
			return

		if os_type == OperatingSystem.WINDOWS and WindowsProactorEventLoopPolicy is not None:
			try:
				asyncio.set_event_loop_policy(WindowsProactorEventLoopPolicy())
				logger.info('Windows event loop policy set for compatibility.')
			except Exception as error:
				logger.warning(f'Failed to set event loop policy: {error}')


platform_service = PlatformService()
