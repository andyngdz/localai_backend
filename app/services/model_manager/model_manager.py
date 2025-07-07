import asyncio
import gc
import logging
from multiprocessing import Process, Queue
from typing import Any, Dict

import torch
from diffusers import AutoPipelineForText2Image

from app.database import get_db
from app.database.crud import add_model
from app.routers.downloads.schemas import DownloadCompletedResponse
from app.routers.websocket import SocketEvents, emit
from app.services.storage import get_model_dir
from config import BASE_CACHE_DIR

from .loader import load_model_process
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
        self.pipe = None
        self.id = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.done_queue = Queue()

        logger.info('ModelManager instance initialized.')

    async def monitor_done_queue(self):
        """Background thread to monitor the done queue for model loading completion."""
        while True:
            id = await asyncio.to_thread(self.done_queue.get)
            logger.info(f'Model download completed for ID: {id}')
            await self.model_done(id)

    async def model_done(self, id: str):
        """Handles the completion of a model download."""

        processes = download_processes.get(id)

        if processes:
            processes.kill()
            del download_processes[id]

        [db] = get_db()
        model_dir = get_model_dir(id)

        add_model(db, id, model_dir)

        await emit(
            SocketEvents.DOWNLOAD_COMPLETED,
            DownloadCompletedResponse(id=id).model_dump(),
        )

        logger.info(f'Model {id} download completed and added to database.')

    def clear_cuda_cache(self):
        """Clears the CUDA cache if available."""

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info('CUDA cache cleared.')
        else:
            logger.warning('CUDA is not available, cannot clear cache.')

    def start_model_download(self, id: str):
        """Start downloading a model in a separate process."""

        load_process = download_processes[id]

        if load_process and load_process.is_alive():
            logger.info(f'Model download already in progress: {id}')
            return

        new_process = Process(
            target=load_model_process, args=(id, self.device, self.done_queue)
        )

        self.unload_model()

        new_process.start()

        self.id = id
        download_processes[id] = new_process

        logger.info(f'Started background model download: {id}')

    def cancel_model_download(self, id: str):
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

            return dict(self.pipe.config)

        self.unload_model()

        try:
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                id,
                cache_dir=BASE_CACHE_DIR,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                use_safetensors=True,
                low_cpu_mem_usage=True,
            )

            self.pipe.to(self.device)

            if self.device == 'cuda':
                self.pipe.enable_model_cpu_offload()
                self.pipe.enable_attention_slicing()

            logger.info(f'Model {id} loaded successfully.')

            self.id = id

            return dict(self.pipe.config)

        except Exception as e:
            self.pipe = None
            self.id = None

            logger.error(f'Failed to load model {id}: {e}', exc_info=True)
            raise

    def unload_model(self):
        """Unloads the current model and frees VRAM."""
        if self.pipe is not None:
            logger.info(f'Unloading model: {self.id}')

            try:
                del self.pipe
                self.pipe = None
                self.id = None
                self.clear_cuda_cache()
                gc.collect()
            except Exception as e:
                logger.warning(f'Error during unload: {e}')

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


model_manager = ModelManager()
