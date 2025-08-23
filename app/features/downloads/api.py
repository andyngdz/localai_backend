import logging
from asyncio import CancelledError

from aiohttp import ClientError
from fastapi import APIRouter
from huggingface_hub import HfApi
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from app.socket import socket_service

from .schemas import (
	DownloadModelRequest,
	DownloadModelResponse,
	DownloadModelStartResponse,
)
from .services import download_service

logger = logging.getLogger(__name__)
downloads = APIRouter(
	prefix='/downloads',
	tags=['downloads'],
)
api = HfApi()


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

	try:
		await socket_service.download_start(DownloadModelStartResponse(id=id))

		await download_service.start(id)

		return DownloadModelResponse(
			id=id,
			message='Download completed',
		)

	except CancelledError:
		logger.warning(f'Download task for id {id} was cancelled')
		raise
	finally:
		logger.info(f'Download task for id {id} completed')
