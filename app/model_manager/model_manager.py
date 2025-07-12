import asyncio
import gc
import logging
from multiprocessing import Process, Queue
from typing import Any, Dict

import torch

from app.database import get_db
from app.database.crud import add_model
from app.routers.downloads.schemas import DownloadCompletedResponse
from app.routers.websocket import SocketEvents, emit
from app.services.storage import get_model_dir

from .constants import default_sample_size
from .model_loader import model_loader
from .schedulers import (
    SCHEDULER_MAPPING,
    SamplerType,
)
from .states import download_processes

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages the active diffusion pipeline and handles background loading with cancellation.
    """

    def __init__(self):
        [db] = get_db()

        self.db = db
        self.pipe = None
        self.id = None
        self.download_queue = Queue()

        logger.info('ModelManager instance initialized.')

    async def monitor_download(self):
        """Background thread to monitor the done queue for model loading completion."""

        logger.info('Monitoring download tasks.')

        while True:
            id = await asyncio.to_thread(self.download_queue.get)
            logger.info(f'Model download completed for ID: {id}')
            await self.download_done(id)

    async def download_done(self, id: str):
        """Handles the completion of a model download."""

        processes = download_processes.get(id)

        if processes:
            processes.kill()
            del download_processes[id]

        model_dir = get_model_dir(id)

        add_model(self.db, id, model_dir)

        await emit(
            SocketEvents.DOWNLOAD_COMPLETED,
            DownloadCompletedResponse(id=id).model_dump(),
        )

        logger.info(f'Model {id} download completed and added to database.')

    def clear_cache(self):
        """Clears the CUDA cache if available."""

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info('CUDA cache cleared.')
        else:
            logger.warning('CUDA is not available, cannot clear cache.')

        logging.info('Forcing garbage collection to free memory.')
        gc.collect()

    def start_download(self, id: str):
        """Start downloading a model in a separate process."""

        download_process = download_processes.get(id)

        if download_process and download_process.is_alive():
            logger.info(f'Model download already in progress: {id}')
            return

        self.unload_model()
        new_process = Process(
            target=model_loader.process, args=(id, self.db, self.download_queue)
        )
        new_process.start()
        download_processes[id] = new_process

        self.id = id
        logger.info(f'Started background model download: {id}')

    def cancel_download(self, id: str):
        """Cancel the active model download and clean up cache."""

        self.unload_model()

        download_process = download_processes[id]

        if download_process and download_process.is_alive():
            logger.info(f'Cancelling model download: {id}')

            download_process.terminate()
            download_process.join()
        else:
            logger.info('No active model download to cancel.')

    def load_model(self, id: str) -> Dict[str, Any]:
        """
        Load a model synchronously into memory for inference.
        Should only be called when model is confirmed downloaded.
        """

        logger.info(f'Attempting to load model: {id}')

        if self.id == id and self.pipe is not None:
            logger.info(f'Model {id} is already loaded.')

            unet_config = self.pipe.unet.config
            if unet_config is not None:
                logger.info(f'UNet config: {unet_config}')

            return dict(self.pipe.config)

        self.unload_model()

        try:
            self.pipe = model_loader.process(id, self.db)

            logger.info(f'Model {id} loaded successfully.')

            self.id = id

            return dict(self.pipe.config)

        except Exception as error:
            self.pipe = None
            self.id = None

            logger.error(f'Failed to load model {id}: {error}')
            raise

    def unload_model(self):
        """Unloads the current model and frees VRAM."""

        try:
            if self.pipe is not None:
                logger.info(f'Unloading model: {self.id}')
                del self.pipe
                self.pipe = None
                self.id = None
        except Exception as error:
            logger.warning(f'Error during unload: {error}')
        finally:
            self.clear_cache()

    def set_sampler(self, sampler: SamplerType):
        """Dynamically sets the sampler for the currently loaded pipeline."""

        if not self.pipe:
            raise ValueError('No model loaded. Cannot set sampler.')

        scheduler = SCHEDULER_MAPPING.get(sampler)

        if not scheduler:
            raise ValueError(f'Unsupported sampler type: {sampler.value}')

        config = self.pipe.scheduler.config
        kwargs = {}

        if sampler in [
            SamplerType.DPM_SOLVER_MULTISTEP_KARRAS,
            SamplerType.DPM_SOLVER_SDE_KARRAS,
        ]:
            kwargs['use_karras_sigmas'] = True

        new_scheduler = scheduler.from_config(config, **kwargs)
        self.pipe.scheduler = new_scheduler

        logger.info(f'Sampler set to: {sampler.value}')

    def get_sample_size(self):
        """Returns the sample size of the model based on its configuration."""

        if not self.pipe:
            raise ValueError('No model loaded. Cannot get sample size.')

        unet_config = self.pipe.unet.config

        if hasattr(unet_config, 'sample_size'):
            sample_size = unet_config.sample_size

            return sample_size
        else:
            return default_sample_size


model_manager = ModelManager()
