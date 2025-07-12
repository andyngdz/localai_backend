import asyncio

import socketio

from app.routers.websocket.schemas import SocketEvents

loop = asyncio.get_event_loop()

sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',
    logger=True,
)
sio_app = socketio.ASGIApp(sio, socketio_path='ws')


async def emit(event: SocketEvents, data: dict):
    await sio.emit(event, data=data)


def emit_from_sync(event: str, data: dict):
    asyncio.run_coroutine_threadsafe(sio.emit(event, data=data), loop=loop)
