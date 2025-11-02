"""Factory for creating platform-specific optimizers."""

import sys

from app.services import logger_service

from .base import PlatformOptimizer
from .darwin import DarwinOptimizer
from .linux import LinuxOptimizer
from .windows import WindowsOptimizer

logger = logger_service.get_logger(__name__, category='PlatformOpt')


def get_optimizer() -> PlatformOptimizer:
	"""Get the appropriate optimizer for the current platform.

	Returns:
		PlatformOptimizer: Platform-specific optimizer instance

	Raises:
		RuntimeError: If platform is not supported
	"""
	platform = sys.platform

	if platform == 'win32':
		logger.info('Using Windows optimizer')
		return WindowsOptimizer()
	elif platform == 'linux':
		logger.info('Using Linux optimizer')
		return LinuxOptimizer()
	elif platform == 'darwin':
		logger.info('Using macOS optimizer')
		return DarwinOptimizer()
	else:
		error_msg = f'Unsupported platform: {platform}'
		logger.error(error_msg)
		raise RuntimeError(error_msg)
