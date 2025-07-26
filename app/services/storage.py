import os
from logging import getLogger

from config import CACHE_DIR, CACHE_LOCK_DIR, GENERATED_IMAGES_DIR

logger = getLogger(__name__)


class StorageService:
	"""
	A service for managing storage paths for models and other resources.
	"""

	def __init__(self):
		self.cache_dir = CACHE_DIR
		self.cache_lock_dir = CACHE_LOCK_DIR

	def init(self):
		"""
		Initialize the storage service by creating necessary directories.
		"""

		os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)

		logger.info('Storage service initialized successfully.')

	def get_model_dir(self, id: str) -> str:
		"""Get the directory path for a model based on its ID."""
		name_serialized = id.replace('/', '--')

		dir = os.path.join(CACHE_DIR, f'models--{name_serialized}')

		return dir

	def get_model_lock_dir(self, id: str) -> str:
		"""Get the full path for a model lock file based on its ID."""
		name_serialized = id.replace('/', '--')

		dir = os.path.join(CACHE_LOCK_DIR, f'models--{name_serialized}')

		return dir


storage_service = StorageService()
