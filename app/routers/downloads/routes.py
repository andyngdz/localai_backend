import logging
from asyncio import CancelledError

from aiohttp import ClientError
from fastapi import APIRouter
from huggingface_hub import HfApi
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from app.model_manager import model_manager_service
from app.services import storage_service
from app.socket import socket_service

from .schemas import (
	DownloadModelRequest,
	DownloadModelResponse,
	DownloadModelStartResponse,
)

logger = logging.getLogger(__name__)
downloads = APIRouter(
	prefix='/downloads',
	tags=['downloads'],
)
api = HfApi()


async def start_downloading(id: str):
	"""Run the download task for the given model ID."""

	try:
		await socket_service.download_start(DownloadModelStartResponse(id=id))

		model_dir = storage_service.get_model_dir(id)

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
async def download(request: DownloadModelRequest):
	"""Initialize a download for the given model ID"""

	id = request.id

	logger.info(f'API Request: Initiating download for id: {id}')

	await start_downloading(id)

	return DownloadModelResponse(
		id=id,
		message='Download completed',
	)
