from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .schemas import SocketEvents, SocketResponse

socket = APIRouter()

connected_clients = set()


@socket.websocket('/ws')
async def websocket_init(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    print('WebSocket connected')

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        print('WebSocket disconnected')


async def emit_events(event: SocketEvents, data: dict):
    message = SocketResponse(event=event, data=data).model_dump()

    for ws in connected_clients.copy():
        try:
            await ws.send_json(message)
        except Exception:
            connected_clients.remove(ws)
