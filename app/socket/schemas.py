from enum import Enum

from pydantic import BaseModel


class SocketEvents(str, Enum):
	"""
	Enum for the state of a model download.
	Using str as base class ensures it serializes to string in JSON.
	"""

	DOWNLOAD_START = 'download_start'
	DOWNLOAD_COMPLETED = 'download_completed'
	IMAGE_GENERATION_EACH_STEP = 'image_generation_each_step'


class SocketResponse(BaseModel):
	"""
	Base response model for WebSocket events.
	"""

	event: SocketEvents

	data: dict
