from enum import Enum

from pydantic import BaseModel


class SocketEvents(str, Enum):
    """
    Enum for the state of a model download.
    Using str as base class ensures it serializes to string in JSON.
    """

    DOWNLOAD_PREPARE = 'download_prepare'
    DOWNLOAD_PROGRESS_UPDATE = 'download_progress_update'
    DOWNLOAD_CANCELED = 'download_cancelled'
    DOWNLOAD_COMPLETED = 'download_completed'
    NEW_MODEL_AVAILABLE = 'new_model_available'


class SocketResponse(BaseModel):
    """
    Base response model for WebSocket events.
    """

    event: SocketEvents

    data: dict
