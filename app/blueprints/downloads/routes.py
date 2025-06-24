import logging
import os
from asyncio import CancelledError, Semaphore, gather
from collections import defaultdict
from typing import Dict

from aiohttp import ClientError, ClientSession, ClientTimeout
from flask import Blueprint, jsonify, request
from huggingface_hub import HfApi, hf_hub_url
from pydantic import ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from app.schemas.core import ErrorResponse, ErrorType
from app.services.storage import get_model_dir
from config import CHUNK_SIZE, MAX_CONCURRENT_DOWNLOADS
from socket_io import socketio

from .schemas import (
    DownloadStatus,
    DownloadStatusResponse,
    DownloadStatusStates,
    HuggingFaceRequest,
    download_statuses,
)

logger = logging.getLogger(__name__)
downloads = Blueprint('downloads', __name__)

api = HfApi()
semaphore = Semaphore(MAX_CONCURRENT_DOWNLOADS)

# Global dict to hold progress
# Example structure:
# {
#   "stable-diffusion-v1-5": {
#       "unet/model.bin": {"downloaded": 1024, "total": 2048},
#       ...
#   },
#   ...
# }
progress_store: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(dict)


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
    def progress_callback(filename, downloaded, total):
        progress_store[id][filename] = {
            'downloaded': downloaded,
            'total': total,
        }

        socketio.emit(
            'download_progress_update',
            {
                'id': id,
                'filename': filename,
                'downloaded': downloaded,
                'total': total,
            },
        )

    return progress_callback


async def download_file(
    session: ClientSession,
    dir: str,
    id: str,
    filename: str,
    progress_callback=None,
):
    url = hf_hub_url(id, filename)
    filepath = os.path.join(dir, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    existing_size = 0
    if os.path.exists(filepath):
        existing_size = os.path.getsize(filepath)

    headers = {}
    if existing_size > 0:
        headers['Range'] = f'bytes={existing_size}-'

    try:
        async with session.get(url, headers=headers) as response:
            if response.status == 416:
                # Already fully downloaded
                return

            response.raise_for_status()

            total = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(filepath, 'wb') as file:
                async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                    file.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback and total > 0:
                        progress_callback(filename, downloaded, total)
    except CancelledError:
        logger.warning('Download was cancelled')
        raise


async def limited_download(
    session: ClientSession,
    dir: str,
    id: str,
    filename: str,
    progress_callback=None,
):
    async with semaphore:
        await download_file(session, dir, id, filename, progress_callback)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_fixed(2),
    retry=retry_if_exception_type((TimeoutError, ClientError)),
)
@downloads.route('/', methods=['POST'])
async def init_download():
    try:
        request_data = HuggingFaceRequest.model_validate_json(request.data)
    except ValidationError:
        return handle_validation_error('Missing id', ErrorType.ValidationError)
    except TypeError:
        return handle_validation_error('Value type is incorrect', ErrorType.TypeError)
    except ValueError:
        return handle_validation_error('Value is not valid', ErrorType.ValueError)

    id = request_data.id

    logger.info('API Request: Initiating download for id: %s', id)

    download_statuses[id] = DownloadStatus(id=id)

    dir = get_model_dir(id)
    os.makedirs(dir, exist_ok=True)

    progress_callback = get_progress_callback(id)
    timeout = ClientTimeout(total=None, sock_read=60)
    files = api.list_repo_files(id)

    async with ClientSession(timeout=timeout) as session:
        tasks = [
            limited_download(
                session,
                dir,
                id,
                filename,
                progress_callback=progress_callback,
            )
            for filename in files
        ]

        await gather(*tasks)

    return (
        jsonify(
            DownloadStatusResponse(
                id=id,
                status=DownloadStatusStates.COMPLETED,
                message='Download completed',
            ).model_dump()
        ),
        202,
    )
