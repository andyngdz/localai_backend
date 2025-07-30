from enum import IntEnum

DEFAULT_MAX_GPU_SCALE_FACTOR = 0.5
DEFAULT_MAX_RAM_SCALE_FACTOR = 0.5

DATABASE_URL = 'sqlite:///localai_backend.db'


class DeviceSelection(IntEnum):
	NOT_FOUND = -2
