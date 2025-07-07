# manager.py
import logging
from multiprocessing import Process
from typing import Any, Dict

import torch

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

        logger.info('ModelManager instance initialized.')

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
            raise RuntimeError('A model is already downloading.')

        new_process = Process(target=load_model_process, args=(id, self.device))
        new_process.start()
        download_processes[id] = new_process

        logger.info(f'Started background model download: {id}')

        return new_process

    def cancel_model_download(self, id: str):
        """Cancel the active model download and clean up cache."""

        load_process = download_processes[id]

        if load_process and load_process.is_alive():
            logger.info(f'Cancelling model download: {id}')

            load_process.terminate()
            load_process.join()
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

        if self.pipe:
            self.unload_model()

        try:
            self.pipe = load_model_process(id, self.device)
            self.id = id

            logger.info(f'Model {id} loaded successfully.')

            return dict(self.pipe.config)

        except Exception as e:
            self.pipe = None
            self.id = None

            logger.error(f'Failed to load model {id}: {e}', exc_info=True)
            raise

    def unload_model(self):
        """Unloads the current model and frees VRAM."""
        if self.pipe:
            logger.info(f'Unloading model: {self.id}')

            try:
                del self.pipe
                self.pipe = None
                self.id = None
                self.clear_cuda_cache()
            except Exception as e:
                logger.warning(f'Error during unload: {e}')
        else:
            logger.info('No model loaded.')

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
