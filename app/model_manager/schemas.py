from typing import Dict, Union


class MaxMemoryConfig:
    def __init__(
        self, device: str, device_index: Union[int, str] = 0, memory_size: str = '8GB'
    ):
        """
        device: 'cuda' or 'cpu'
        device_index: index of the CUDA device (default 0)
        memory_size: string representation of memory (default "8GB")
        """
        self.device = device
        self.device_index = device_index
        self.memory_size = memory_size

    def to_dict(self) -> Dict[Union[int, str], str]:
        if self.device == 'cuda':
            return {self.device_index: self.memory_size, 'cpu': self.memory_size}
        else:
            return {'cpu': self.memory_size}
