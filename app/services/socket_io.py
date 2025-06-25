from enum import Enum

from flask_socketio import SocketIO


class SocketEvents(str, Enum):
    """
    Enum for the state of a model download.
    Using str as base class ensures it serializes to string in JSON.
    """

    DOWNLOAD_PROGRESS_UPDATE = 'download_progress_update'


socketio = SocketIO()
