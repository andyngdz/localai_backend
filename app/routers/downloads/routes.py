import logging
import os
import shutil
from asyncio import CancelledError

from aiohttp import ClientError
from fastapi import APIRouter, Query
from huggingface_hub import HfApi
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from app.routers.websocket import SocketEvents, emit
from app.services.model_manager import model_manager
from app.services.storage import get_model_dir, get_model_lock_dir

from .schemas import (
    DownloadCancelledResponse,
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


def delete_model_from_cache(id: str):
    """Clean up the model directory if it exists"""

    model_dir = get_model_dir(id)
    model_lock_dir = get_model_lock_dir(id)

    if os.path.exists(model_dir):
        logger.info('Cleaning up model directory: %s', model_dir)
        shutil.rmtree(model_dir)

    if os.path.exists(model_lock_dir):
        logger.info('Cleaning up model lock directory: %s', model_lock_dir)
        shutil.rmtree(model_lock_dir)


async def clean_up(id: str):
    """Clean up the model directory and remove task from tracking"""

    delete_model_from_cache(id)

    await emit(
        SocketEvents.DOWNLOAD_CANCELED,
        DownloadCancelledResponse(id=id).model_dump(),
    )

    logger.info('Cleaned up resources for id: %s', id)


async def run_download(id: str):
    """Run the download task for the given model ID."""

    try:
        await emit(
            SocketEvents.DOWNLOAD_START,
            DownloadStartResponse(id=id).model_dump(),
        )

        model_dir = get_model_dir(id)

        logger.info('Download model into folder: %s', model_dir)

        model_manager.start_model_download(id)

    except CancelledError:
        logger.warning('Download task for id %s was cancelled', id)
    finally:
        logger.info('Download task for id %s completed', id)


@downloads.post('/init', response_model=DownloadStatusResponse)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_fixed(2),
    retry=retry_if_exception_type((TimeoutError, ClientError)),
)
async def init_download(request: DownloadRequest):
    """Initialize a download for the given model ID"""

    id = request.id

    logger.info('API Request: Initiating download for id: %s', id)

    await run_download(id)

    return DownloadStatusResponse(
        id=id,
        message='Download completed',
    )


@downloads.get('/cancel', response_model=DownloadStatusResponse)
async def cancel_download(id: str = Query(..., description='The model ID to cancel')):
    """Cancel the download by id"""

    logger.info('API Request: Cancelling download for id: %s', id)

    model_manager.cancel_model_download(id)
    await clean_up(id)

    return DownloadStatusResponse(
        id=id,
        message='Download cancelled',
    )
