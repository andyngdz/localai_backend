import logging
from asyncio import CancelledError

from aiohttp import ClientError
from fastapi import APIRouter
from huggingface_hub import HfApi
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
import thread

from app.model_manager.model_manager_service import model_manager_service
from app.services import get_model_dir
from app.socket import SocketEvents, socket_service

from .schemas import (
    DownloadRequest,
    DownloadStartResponse,
    DownloadStatusResponse,
)

logger = logging.getLogger(__name__)
downloads = APIRouter(
    prefix='/downloads',
    tags=['downloads'],
)
api = HfApi()


async def run_download(id: str):
    """Run the download task for the given model ID."""

    try:
        await socket_service.emit(
            SocketEvents.DOWNLOAD_START,
            DownloadStartResponse(id=id).model_dump(),
        )

        model_dir = get_model_dir(id)

        logger.info(f'Download model into folder: {model_dir}')

        await model_manager_service.load_model_async(id)

    except CancelledError:
        logger.warning(f'Download task for id {id} was cancelled')
    finally:
        logger.info(f'Download task for id {id} completed')


@downloads.post('/')
@retry(
    stop=stop_after_attempt(5),
    wait=wait_fixed(2),
    retry=retry_if_exception_type((TimeoutError, ClientError)),
)
async def init_download(request: DownloadRequest):
    """Initialize a download for the given model ID"""

    id = request.id

    logger.info(f'API Request: Initiating download for id: {id}')

    await run_download(id)

    return DownloadStatusResponse(
        id=id,
        message='Download completed',
    )
