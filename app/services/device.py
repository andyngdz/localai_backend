import platform

import torch

from app.services.logger import logger_service

logger = logger_service.get_logger(__name__, category='Service')


class DeviceService:
	"""
	Service to manage device-related operations.
	"""

	def __init__(self):
		# Priority: CUDA > MPS > CPU
		try:
			if torch.cuda.is_available():
				self.device = 'cuda'
				self.torch_dtype = torch.float16
				logger.info(f'CUDA device detected: {torch.cuda.get_device_name(0)}')
			elif torch.backends.mps.is_available():
				self.device = 'mps'
				# Use float32 for MPS to avoid numerical instability issues that cause NaN values
				self.torch_dtype = torch.float32
				logger.info(f'MPS device detected: Apple {platform.machine()}')
			else:
				self.device = 'cpu'
				self.torch_dtype = torch.float32
				logger.info('Using CPU device (no GPU acceleration available)')
		except Exception as error:
			# Fallback to CPU if there are any issues with device detection
			logger.warning(f'Device detection failed, falling back to CPU: {error}')
			self.device = 'cpu'
			self.torch_dtype = torch.float32

	def get_device_name(self, index: int) -> str:
		if self.is_cuda:
			try:
				return torch.cuda.get_device_name(index)
			except Exception as error:
				logger.error(f'Failed to get CUDA device name for index {index}: {error}')
				return f'cuda:{index}'
		elif self.is_mps:
			# For MPS, return the Apple Silicon chip name
			return f'Apple {platform.machine()}'

		return 'cpu'

	def get_device_properties(self, index: int):
		if self.is_cuda:
			try:
				return torch.cuda.get_device_properties(index)
			except Exception as error:
				logger.error(f'Failed to get CUDA device properties for index {index}: {error}')
				return None

		# MPS and CPU don't have device properties like CUDA
		return None

	@property
	def is_cuda(self):
		return self.device == 'cuda'

	@property
	def is_mps(self):
		return self.device == 'mps'

	@property
	def current_device(self):
		if self.is_cuda:
			return torch.cuda.current_device()
		# MPS and CPU don't have multiple devices
		return 0

	@property
	def device_count(self):
		if self.is_cuda:
			return torch.cuda.device_count()
		elif self.is_mps:
			# MPS typically has 1 device (the Apple Silicon GPU)
			return 1
		return 0

	@property
	def is_available(self):
		if self.is_cuda:
			return torch.cuda.is_available()
		elif self.is_mps:
			return torch.backends.mps.is_available()
		return False

	def get_recommended_batch_size(self) -> int:
		"""
		Calculate recommended maximum batch size based on available GPU memory.

		Returns:
			int: Recommended maximum number of images to generate in a single batch

		Memory tiers:
			- < 4GB: 1 image
			- 4-8GB: 2 images
			- 8-12GB: 3 images
			- 12-16GB: 4 images
			- >= 16GB: 6 images
		"""
		from app.cores.constants.batch_size import BATCH_SIZE_THRESHOLDS

		if not self.is_available:
			return 1  # CPU only - be conservative

		# Get total GPU memory in GB
		if self.is_cuda:
			props = self.get_device_properties(0)
			if props:
				total_memory_gb = props.total_memory / (1024**3)
			else:
				return 1
		elif self.is_mps:
			# MPS shares memory between CPU and GPU, be more conservative
			return 2
		else:
			return 1

		# Find appropriate threshold based on available memory
		for memory_threshold, batch_size in BATCH_SIZE_THRESHOLDS:
			if total_memory_gb < memory_threshold:
				return batch_size

		# Fallback to largest batch size if memory exceeds all thresholds
		return BATCH_SIZE_THRESHOLDS[-1][1]


device_service = DeviceService()
