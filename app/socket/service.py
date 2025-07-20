import asyncio
from logging import getLogger

import socketio

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
		self.sio_app = socketio.ASGIApp(self.sio, socketio_path='ws')

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


socket_service = SocketService()
