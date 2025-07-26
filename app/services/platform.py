import asyncio
import sys
from logging import getLogger

logger = getLogger(__name__)


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
