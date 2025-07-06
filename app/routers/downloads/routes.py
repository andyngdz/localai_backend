import logging
import os
import shutil
from asyncio import CancelledError, Semaphore

from aiohttp import ClientError
from fastapi import APIRouter, Depends, Query
from huggingface_hub import HfApi
from sqlalchemy.orm import Session
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from app.database import get_db
from app.database.crud import add_model
from app.routers.websocket import SocketEvents, emit
from app.services import get_model_dir, model_manager
from app.services.storage import get_locks_model_dir
from config import MAX_CONCURRENT_DOWNLOADS

from .schemas import (
    DownloadCancelledResponse,
    DownloadCompletedResponse,
    DownloadRequest,
    DownloadStartResponse,
    DownloadStatusResponse,
)

logger = logging.getLogger(__name__)
downloads = APIRouter(
    prefix='/downloads',
    tags=['downloads'],
)
semaphore = Semaphore(MAX_CONCURRENT_DOWNLOADS)
api = HfApi()


def delete_model_from_disk(id: str):
    """Clean up the model directory if it exists"""

    model_dir = get_model_dir(id)
    model_lock_dir = get_locks_model_dir(id)

    if os.path.exists(model_dir):
        logger.info('Cleaning up model directory: %s', model_dir)
        shutil.rmtree(model_dir)
        shutil.rmtree(model_lock_dir)


async def clean_up(id: str):
    """Clean up the model directory and remove task from tracking"""

    delete_model_from_disk(id)

    await emit(
        SocketEvents.DOWNLOAD_CANCELED, DownloadCancelledResponse(id=id).model_dump()
    )

    logger.info('Cleaned up resources for id: %s', id)


async def run_download(id: str, db: Session):
    """Run the download task for the given model ID."""

    try:
        await emit(
            SocketEvents.DOWNLOAD_START,
            DownloadStartResponse(id=id).model_dump(),
        )

        model_dir = get_model_dir(id)

        logger.info('Download model into folder: %s', model_dir)

        process = model_manager.start_model_download(id)

        process.join()

        await emit(
            SocketEvents.DOWNLOAD_COMPLETED,
            DownloadCompletedResponse(id=id).model_dump(),
        )

        add_model(db, id, model_dir)
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
async def init_download(request: DownloadRequest, db: Session = Depends(get_db)):
    """Initialize a download for the given model ID"""

    id = request.id

    logger.info('API Request: Initiating download for id: %s', id)

    await run_download(id, db)

    return DownloadStatusResponse(
        id=id,
        message='Download completed',
    )


@downloads.get('/cancel', response_model=DownloadStatusResponse)
async def cancel_download(id: str = Query(..., description='The model ID to cancel')):
    """Cancel the download by id"""

    logger.info('API Request: Cancelling download for id: %s', id)

    model_manager.cancel_model_download()
    await clean_up(id)

    return DownloadStatusResponse(
        id=id,
        message='Download cancelled',
    )
