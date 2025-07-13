import asyncio
import logging
from multiprocessing import Process, Queue

from sqlalchemy.orm import Session

from app.model_loader import model_loader
from app.database.crud import add_model
from app.services import get_model_dir
from app.socket import SocketEvents, socket_service

from .schemas import DownloadCompletedResponse
from .states import download_processes

logger = logging.getLogger(__name__)


class ModelDownloadService:
    """Manages the background downloading of models and their loading into memory."""

    def __init__(self):
        self.download_queue = Queue()

        logger.info('ModelDownloadService instance initialized.')

    def start(self, db: Session):
        """Starts the background task to monitor model downloads."""

        logger.info('Starting model download monitoring task.')
        self.db = db
        asyncio.create_task(self.monitor_download())

    async def monitor_download(self):
        """Background thread to monitor the done queue for model loading completion."""

        while True:
            id = await asyncio.to_thread(self.download_queue.get)
            if id:
                logger.info(f'Model download completed for ID: {id}')
                await self.download_completed(id)

    async def download_completed(self, id: str):
        """Handles the completion of a model download."""

        processes = download_processes.get(id)

        if processes:
            processes.kill()
            del download_processes[id]

        model_dir = get_model_dir(id)

        add_model(self.db, id, model_dir)

        await socket_service.emit(
            SocketEvents.DOWNLOAD_COMPLETED,
            DownloadCompletedResponse(id=id).model_dump(),
        )

        logger.info(f'Model {id} download completed and added to database.')

    def start_download(self, id: str):
        """Start downloading a model in a separate process."""

        download_process = download_processes.get(id)

        if download_process and download_process.is_alive():
            logger.info(f'Model download already in progress: {id}')
            return

        new_process = Process(
            target=model_loader,
            args=(id, self.download_queue),
        )
        new_process.start()
        download_processes[id] = new_process

        logger.info(f'Started background model download: {id}')

    def cancel_download(self, id: str):
        """Cancel the active model download and clean up cache."""

        download_process = download_processes.get(id)

        if download_process and download_process.is_alive():
            logger.info(f'Cancelling model download: {id}')

            download_process.terminate()
            download_process.join()
        else:
            logger.info('No active model download to cancel.')


model_download_service = ModelDownloadService()
