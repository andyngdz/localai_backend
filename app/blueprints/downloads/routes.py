import logging
import os
import shutil
from asyncio import CancelledError, Semaphore, create_task, gather

import aiofiles
from aiohttp import ClientError, ClientSession, ClientTimeout
from flask import Blueprint, jsonify, request
from huggingface_hub import HfApi, hf_hub_url
from pydantic import ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from app.schemas.core import ErrorResponse, ErrorType
from app.services.socket_io import SocketEvents, socketio
from app.services.storage import get_model_dir
from config import CHUNK_SIZE, MAX_CONCURRENT_DOWNLOADS

from .schemas import (
    DownloadProgressResponse,
    DownloadRequest,
    DownloadStatusResponse,
)
from .states import download_progresses, download_tasks

logger = logging.getLogger(__name__)
downloads = Blueprint('downloads', __name__)
semaphore = Semaphore(MAX_CONCURRENT_DOWNLOADS)
api = HfApi()


def delete_model_from_disk(id: str):
    """Clean up the model directory if it exists"""
    model_dir = get_model_dir(id)

    if os.path.exists(model_dir):
        logger.info('Cleaning up model directory: %s', model_dir)
        shutil.rmtree(model_dir)


def clean_up(id: str):
    """Clean up the model directory and remove task from tracking"""

    delete_model_from_disk(id)
    download_tasks.pop(id, None)
    download_progresses.pop(id, None)
    logger.info('Cleaned up resources for id: %s', id)


def handle_validation_error(detail: str, type: ErrorType):
    logger.error('%s for download request: %s', type, detail)

    return (
        jsonify(
            ErrorResponse(
                detail=detail,
                type=type,
            ).model_dump()
        ),
        400,
    )


def get_progress_callback(id: str):
    def progress_callback(filename: str, downloaded: int, total: int):
        download_progresses[id][filename] = {
            'downloaded': downloaded,
            'total': total,
        }

        socketio.emit(
            SocketEvents.DOWNLOAD_PROGRESS_UPDATE,
            DownloadProgressResponse(
                id=id,
                filename=filename,
                downloaded=downloaded,
                total=total,
            ).model_dump(),
        )

    return progress_callback


async def run_download(id: str):
    try:
        files = api.list_repo_files(id)
        model_dir = get_model_dir(id)
        os.makedirs(model_dir, exist_ok=True)
        progress_callback = get_progress_callback(id)
        timeout = ClientTimeout(total=None, sock_read=60)

        async with ClientSession(timeout=timeout) as session:
            tasks = [
                limited_download(
                    session,
                    model_dir,
                    id,
                    filename,
                    progress_callback,
                )
                for filename in files
            ]

            await gather(*tasks)
    except CancelledError:
        logger.warning('Download task for id %s was cancelled', id)
    finally:
        clean_up(id)
        logger.info('Download task for id %s completed', id)


async def download_file(
    session: ClientSession,
    model_dir: str,
    id: str,
    filename: str,
    progress_callback,
):
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
            # Check if already fully downloaded, then skip
            if response.status == 416:
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
                        progress_callback(filename, downloaded, total)
    except CancelledError:
        logger.warning('Download %s was cancelled', filename)
        raise


async def limited_download(
    session: ClientSession,
    model_dir: str,
    id: str,
    filename: str,
    progress_callback,
):
    async with semaphore:
        await download_file(session, model_dir, id, filename, progress_callback)


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


@retry(
    stop=stop_after_attempt(5),
    wait=wait_fixed(2),
    retry=retry_if_exception_type((TimeoutError, ClientError)),
)
@downloads.route('/', methods=['POST'])
async def init_download():
    try:
        request_data = DownloadRequest.model_validate_json(request.data)
    except ValidationError:
        return handle_validation_error('Missing id', ErrorType.ValidationError)
    except TypeError:
        return handle_validation_error('Value type is incorrect', ErrorType.TypeError)
    except ValueError:
        return handle_validation_error('Value is not valid', ErrorType.ValueError)

    id = request_data.id

    logger.info('API Request: Initiating download for id: %s', id)

    if id in download_tasks and not download_tasks[id].done():
        await cancel_task(id)

    task = create_task(run_download(id))
    download_tasks[id] = task
    await task

    return (
        jsonify(
            DownloadStatusResponse(
                id=id,
                message='Download completed',
            ).model_dump()
        ),
        202,
    )


@downloads.route('/cancel', methods=['GET'])
async def cancel_download():
    """Cancel the download by id"""
    id = request.args.get('id')

    if not id:
        return handle_validation_error(
            "Missing 'id' query parameter", ErrorType.ValidationError
        )

    await cancel_task(id)

    return jsonify(
        DownloadStatusResponse(
            id=id,
            message='Download cancelled',
        ).model_dump()
    ), 200
