from .device import device_service
from .image import image_service
from .logger import logger_service
from .memory import MemoryService
from .platform import platform_service
from .recommendations import ModelRecommendationService
from .storage import storage_service
from .styles import styles_service

__all__ = [
	'MemoryService',
	'platform_service',
	'device_service',
	'ModelRecommendationService',
	'logger_service',
	'styles_service',
	'image_service',
	'storage_service',
]
