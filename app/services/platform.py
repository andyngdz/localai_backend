import asyncio
import sys

from app.services.logger import logger_service

logger = logger_service.get_logger(__name__, category='Service')


class PlatformService:
	"""Service to handle platform-specific operations."""

	def init(self):
		"""Initialize the platform service."""

		if sys.platform == 'win32':
			try:
				asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
				logger.info('Windows event loop policy set for compatibility.')
			except Exception as error:
				logger.warning(f'Failed to set event loop policy: {error}')


platform_service = PlatformService()
