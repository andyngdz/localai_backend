import socketio

from app.routers.websocket.schemas import SocketEvents

socket_server = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',
)
socket_app = socketio.ASGIApp(socket_server, socketio_path='ws')


async def emit(event: SocketEvents, data: dict):
    await socket_server.emit(event, data=data)
