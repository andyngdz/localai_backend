import asyncio
from logging import getLogger

import socketio
from pydantic import BaseModel

from .schemas import SocketEvents

logger = getLogger(__name__)


class SocketService:
	"""
	Service for handling socket events.
	"""

	def __init__(self):
		self.loop = asyncio.get_event_loop()
		self.sio = socketio.AsyncServer(
			async_mode='asgi',
			cors_allowed_origins='*',
			logger=True,
		)
		self.sio_app = socketio.ASGIApp(self.sio)

		logger.info('SocketService initialized.')

	async def emit(self, event: SocketEvents, data: dict):
		await self.sio.emit(event, data=data)

	def emit_sync(self, event: SocketEvents, data: dict):
		"""
		Emit an event synchronously to all connected clients.
		"""
		asyncio.run_coroutine_threadsafe(
			self.sio.emit(event, data=data),
			loop=self.loop,
		)

	async def download_start(self, data: BaseModel):
		"""
		Emit a download start event with the provided data.
		"""
		await self.emit(SocketEvents.DOWNLOAD_START, data=data.model_dump())

	async def download_completed(self, data: BaseModel):
		"""
		Emit a download completed event with the provided data.
		"""
		await self.emit(SocketEvents.DOWNLOAD_COMPLETED, data=data.model_dump())

	def download_step_progress(self, data: BaseModel):
		"""
		Emit a download step progress event synchronously with the provided data.
		"""
		self.emit_sync(SocketEvents.DOWNLOAD_STEP_PROGRESS, data=data.model_dump())

	def model_load_completed(self, data: BaseModel):
		"""
		Emit a model load completed event with the provided data.
		"""
		self.emit_sync(SocketEvents.MODEL_LOAD_COMPLETED, data=data.model_dump())

	def image_generation_step_end(self, data: BaseModel):
		"""
		Emit an image generation step end event with the provided data.
		"""
		self.emit_sync(SocketEvents.IMAGE_GENERATION_STEP_END, data=data.model_dump())


socket_service = SocketService()
