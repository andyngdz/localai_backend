import socketio

from app.routers.websocket.schemas import SocketEvents

app_socket_server = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',
    logger=True,
)
app_socket = socketio.ASGIApp(app_socket_server, socketio_path='ws')


async def emit(event: SocketEvents, data: dict):
    await app_socket_server.emit(event, data=data)
