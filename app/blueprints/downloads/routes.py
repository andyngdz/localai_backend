import logging
import os
from asyncio import Semaphore, gather
from collections import defaultdict
from typing import Dict

from aiohttp import ClientSession
from flask import Blueprint, jsonify, request
from huggingface_hub import HfApi, hf_hub_url
from pydantic import ValidationError

from app.schemas.core import ErrorResponse, ErrorType
from socket_io import socketio

from .schemas import (
    DownloadStatus,
    DownloadStatusResponse,
    DownloadStatusStates,
    HuggingFaceRequest,
    download_statuses,
)

MAX_CONCURRENT_DOWNLOADS = 8
CHUNK_SIZE = 8192
BASE_MODEL_DIR = './.models'

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


def get_progress_callback(model_id: str):
    def progress_callback(filename, downloaded, total):
        progress_store[model_id][filename] = {
            'downloaded': downloaded,
            'total': total,
        }

        socketio.emit(
            'download_progress_update',
            {
                'model_id': model_id,
                'filename': filename,
                'downloaded': downloaded,
                'total': total,
            },
        )

    return progress_callback


async def download_file(
    session: ClientSession,
    model_dir: str,
    model_id: str,
    filename: str,
    progress_callback=None,
):
    url = hf_hub_url(model_id, filename)
    filepath = os.path.join(model_dir, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    async with session.get(url) as response:
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(filepath, 'wb') as file:
            async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                file.write(chunk)
                downloaded += len(chunk)
                if progress_callback and total > 0:
                    progress_callback(filename, downloaded, total)


async def limited_download(
    session: ClientSession,
    model_dir: str,
    model_id: str,
    filename: str,
    progress_callback=None,
):
    async with semaphore:
        await download_file(session, model_dir, model_id, filename, progress_callback)


@downloads.route('/', methods=['POST'])
async def init_download():
    try:
        request_data = HuggingFaceRequest.model_validate_json(request.data)
    except ValidationError:
        return handle_validation_error('Missing model_id', ErrorType.ValidationError)
    except TypeError:
        return handle_validation_error('Value type is incorrect', ErrorType.TypeError)
    except ValueError:
        return handle_validation_error('Value is not valid', ErrorType.ValueError)

    model_id = request_data.model_id

    logger.info('API Request: Initiating download for model_id: %s', model_id)

    download_statuses[model_id] = DownloadStatus(model_id=model_id)

    model_dir = os.path.join(BASE_MODEL_DIR, model_id.replace('/', '--'))
    os.makedirs(model_dir, exist_ok=True)

    progress_callback = get_progress_callback(model_id)
    files = api.list_repo_files(model_id)
    async with ClientSession() as session:
        tasks = [
            limited_download(
                session,
                model_dir,
                model_id,
                filename,
                progress_callback=progress_callback,
            )
            for filename in files
        ]

        await gather(*tasks)

    return (
        jsonify(
            DownloadStatusResponse(
                model_id=model_id,
                status=DownloadStatusStates.COMPLETED,
                message='Download completed',
            ).model_dump()
        ),
        202,
    )
