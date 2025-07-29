import torch


class DeviceService:
	"""
	Service to manage device-related operations.
	"""

	def __init__(self):
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.torch_dtype = torch.float16 if self.device == 'cuda' else torch.float32

	def get_device_name(self, index: int) -> str:
		if self.is_cuda:
			return torch.cuda.get_device_name(index)

		return 'cpu'

	def get_device_properties(self, index: int):
		if self.is_cuda:
			return torch.cuda.get_device_properties(index)

		return None

	@property
	def is_cuda(self):
		return self.device == 'cuda'

	@property
	def current_device(self):
		return torch.cuda.current_device() if self.is_cuda else -1

	@property
	def device_count(self):
		return torch.cuda.device_count() if self.is_cuda else 0

	@property
	def is_available(self):
		return torch.cuda.is_available()


device_service = DeviceService()
