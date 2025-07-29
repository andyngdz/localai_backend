import platform

import torch


class DeviceService:
	"""
	Service to manage device-related operations.
	"""

	def __init__(self):
		# Priority: CUDA > MPS > CPU
		if torch.cuda.is_available():
			self.device = 'cuda'
			self.torch_dtype = torch.float16
		elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
			self.device = 'mps'
			# Use float32 for MPS to avoid numerical instability issues that cause NaN values
			self.torch_dtype = torch.float32
		else:
			self.device = 'cpu'
			self.torch_dtype = torch.float32

	def get_device_name(self, index: int) -> str:
		if self.is_cuda:
			return torch.cuda.get_device_name(index)
		elif self.is_mps:
			# For MPS, return the Apple Silicon chip name
			return f'Apple {platform.machine()}'

		return 'cpu'

	def get_device_properties(self, index: int):
		if self.is_cuda:
			return torch.cuda.get_device_properties(index)

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


device_service = DeviceService()
