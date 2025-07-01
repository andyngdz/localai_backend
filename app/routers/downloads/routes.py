import logging
import os
import shutil
from asyncio import CancelledError, Semaphore, create_task, gather

import aiofiles
from aiohttp import ClientError, ClientSession, ClientTimeout
from fastapi import APIRouter, Depends, Query, status
from huggingface_hub import HfApi, hf_hub_url
from sqlalchemy.orm import Session
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from app.database import get_db
from app.database.crud import add_model
from app.routers.downloads.types import ProgressCallbackType
from app.routers.websocket import SocketEvents, emit
from app.services import get_model_dir
from config import CHUNK_SIZE, MAX_CONCURRENT_DOWNLOADS

from .schemas import (
    DownloadCancelledResponse,
    DownloadCompletedResponse,
    DownloadPrepareResponse,
    DownloadProgressResponse,
    DownloadRequest,
    DownloadStatusResponse,
)
from .states import download_progresses, download_tasks

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

    if os.path.exists(model_dir):
        logger.info('Cleaning up model directory: %s', model_dir)
        shutil.rmtree(model_dir)


def get_progress_callback(id: str) -> ProgressCallbackType:
    """Create a progress callback function for tracking download progress."""

    async def progress_callback(filename: str, downloaded: int, total: int):
        download_progresses[id][filename] = {
            'downloaded': downloaded,
            'total': total,
        }

        await emit(
            SocketEvents.DOWNLOAD_PROGRESS_UPDATE,
            DownloadProgressResponse(
                id=id,
                filename=filename,
                downloaded=downloaded,
                total=total,
            ).model_dump(),
        )

    return progress_callback


async def clean_up(id: str):
    """Clean up the model directory and remove task from tracking"""

    delete_model_from_disk(id)
    download_tasks.pop(id, None)
    download_progresses.pop(id, None)
    await emit(
        SocketEvents.DOWNLOAD_CANCELED, DownloadCancelledResponse(id=id).model_dump()
    )
    logger.info('Cleaned up resources for id: %s', id)


async def run_download(id: str, db: Session):
    """Run the download task for the given model ID."""

    try:
        files = api.list_repo_files(id)
        model_dir = get_model_dir(id)
        progress_callback = get_progress_callback(id)
        timeout = ClientTimeout(total=None, sock_read=60)

        await emit(
            SocketEvents.DOWNLOAD_PREPARE,
            DownloadPrepareResponse(
                id=id,
                files=files,
            ).model_dump(),
        )

        async with ClientSession(timeout=timeout) as session:
            tasks = [
                limited_download(session, model_dir, id, filename, progress_callback)
                for filename in files
            ]

            await gather(*tasks)

        await emit(
            SocketEvents.DOWNLOAD_COMPLETED,
            DownloadCompletedResponse(id=id).model_dump(),
        )
        add_model(db, id, model_dir)
    except CancelledError:
        await clean_up(id)
        logger.warning('Download task for id %s was cancelled', id)
    finally:
        logger.info('Download task for id %s completed', id)


async def limited_download(
    session: ClientSession,
    model_dir: str,
    id: str,
    filename: str,
    progress_callback: ProgressCallbackType,
):
    """Download a file with semaphore to limit concurrent downloads."""

    async with semaphore:
        await download_file(session, model_dir, id, filename, progress_callback)


async def download_file(
    session: ClientSession,
    model_dir: str,
    id: str,
    filename: str,
    progress_callback: ProgressCallbackType,
):
    """Download a single file from the Hugging Face Hub."""

    url = hf_hub_url(id, filename)
    filepath = os.path.join(model_dir, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    existing_size = 0
    if os.path.exists(filepath):
        existing_size = os.path.getsize(filepath)

    headers = {}
    if existing_size > 0:
        headers['Range'] = f'bytes={existing_size}-'

    try:
        async with session.get(url, headers=headers) as response:
            if response.status == status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE:
                return

            response.raise_for_status()

            total = int(response.headers.get('content-length', 0))
            downloaded = 0

            logger.info('Starting download for %s (id: %s)', filename, id)

            async with aiofiles.open(filepath, 'wb') as file:
                async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                    await file.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback and total > 0:
                        await progress_callback(filename, downloaded, total)
    except CancelledError:
        logger.warning('Download %s was cancelled', filename)
        raise


async def cancel_task(id: str):
    """Cancel the download task by id"""

    if id in download_tasks:
        task = download_tasks[id]

        if not task.done():
            try:
                task.cancel()
            except CancelledError:
                logger.info('Download task %s was cancelled', id)
        else:
            logger.info('Download task %s is already completed', id)
    else:
        logger.warning('No download task found for id: %s', id)


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

    if id in download_tasks and not download_tasks[id].done():
        return DownloadStatusResponse(
            id=id,
            message='Download already started',
        ), status.HTTP_202_ACCEPTED

    task = create_task(run_download(id, db))
    download_tasks[id] = task

    return DownloadStatusResponse(
        id=id,
        message='Download completed',
    )


@downloads.get('/cancel', response_model=DownloadStatusResponse)
async def cancel_download(id: str = Query(..., description='The model ID to cancel')):
    """Cancel the download by id"""

    logger.info('API Request: Cancelling download for id: %s', id)

    await cancel_task(id)

    return DownloadStatusResponse(
        id=id,
        message='Download cancelled',
    )
