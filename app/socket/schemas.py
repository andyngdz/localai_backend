from enum import Enum

from pydantic import BaseModel


class SocketEvents(str, Enum):
	"""
	Enum for the state of a model download.
	Using str as base class ensures it serializes to string in JSON.
	"""

	DOWNLOAD_START = 'download_start'
	DOWNLOAD_STEP_PROGRESS = 'download_step_progress'
	DOWNLOAD_COMPLETED = 'download_completed'

	MODEL_LOAD_COMPLETED = 'model_load_completed'
	MODEL_LOAD_FAILED = 'model_load_failed'
	IMAGE_GENERATION_STEP_END = 'image_generation_step_end'


class SocketResponse(BaseModel):
	"""
	Base response model for WebSocket events.
	"""

	event: SocketEvents

	data: dict
