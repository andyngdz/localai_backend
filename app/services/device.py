import torch


class DeviceService:
    """
    Service to manage device-related operations.
    """

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.torch_dtype = torch.float16 if self.device == 'cuda' else torch.float32

    @property
    def is_cuda(self):
        return self.device == 'cuda'


device_service = DeviceService()
