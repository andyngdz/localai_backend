# manager.py
import logging
from multiprocessing import Process
from typing import Any, Dict

import torch
from diffusers import AutoPipelineForText2Image

from config import BASE_CACHE_DIR

from .loader import load_model_process
from .schedulers import (
    SCHEDULER_DESCRIPTIONS,
    SCHEDULER_MAPPING,
    SCHEDULER_NAMES,
    SamplerType,
)
from .schemas import AvailableSampler

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages the active diffusion pipeline and handles background loading with cancellation.
    """

    def __init__(self):
        self.pipe = None
        self.current_id = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_process = None
        self.load_id = None

        logger.info('ModelManager instance initialized.')

    def start_model_download(self, id: str):
        """Start downloading a model in a separate process."""
        if self.load_process and self.load_process.is_alive():
            raise RuntimeError('A model is already downloading.')

        self.load_id = id
        self.load_process = Process(
            target=load_model_process,
            args=(id, self.device, BASE_CACHE_DIR),
        )
        self.load_process.start()

        logger.info(f'Started background model download: {id}')

        return self.load_process

    def cancel_model_download(self):
        """Cancel the active model download and clean up cache."""

        if self.load_process and self.load_process.is_alive():
            logger.info(f'Cancelling model download: {self.load_id}')

            self.load_process.terminate()
            self.load_process.join()
        else:
            logger.info('No active model download to cancel.')

        self.load_process = None
        self.load_id = None

    def load_model(self, id: str) -> Dict[str, Any]:
        """
        Load a model synchronously into memory for inference.
        Should only be called when model is confirmed downloaded.
        """
        logger.info(f'Attempting to load model: {id}')

        if self.current_id == id and self.pipe is not None:
            logger.info(f'Model {id} is already loaded.')
            return dict(self.pipe.config)

        if self.pipe:
            self.unload_model()

        try:
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                id,
                cache_dir=BASE_CACHE_DIR,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                use_safetensors=True,
            )

            self.pipe.to(self.device)

            if self.device == 'cuda':
                self.pipe.enable_model_cpu_offload()
                self.pipe.enable_attention_slicing()

            self.current_id = id
            logger.info(f'Model {id} loaded successfully.')
            return dict(self.pipe.config)

        except Exception as e:
            self.pipe = None
            self.current_id = None
            logger.error(f'Failed to load model {id}: {e}', exc_info=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

    def unload_model(self):
        """Unloads the current model and frees VRAM."""
        if self.pipe:
            logger.info(f'Unloading model: {self.current_id}')
            try:
                del self.pipe
                self.pipe = None
                self.current_id = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info('CUDA cache emptied.')
            except Exception as e:
                logger.warning(f'Error during unload: {e}')
        else:
            logger.info('No model loaded.')

    def set_sampler(self, sampler_type: SamplerType):
        """Dynamically sets the sampler for the currently loaded pipeline."""
        if not self.pipe:
            raise ValueError('No model loaded. Cannot set sampler.')

        scheduler_class = SCHEDULER_MAPPING.get(sampler_type)
        if not scheduler_class:
            raise ValueError(f'Unsupported sampler type: {sampler_type.value}')

        scheduler_config = self.pipe.scheduler.config
        kwargs = {}
        if sampler_type in [
            SamplerType.DPM_SOLVER_MULTISTEP_KARRAS,
            SamplerType.DPM_SOLVER_SDE_KARRAS,
        ]:
            kwargs['use_karras_sigmas'] = True

        new_scheduler = scheduler_class.from_config(scheduler_config, **kwargs)
        self.pipe.scheduler = new_scheduler
        logger.info(f'Sampler set to: {sampler_type.value}')

    def get_available_samplers(self) -> list[AvailableSampler]:
        """Returns list of supported samplers."""
        return [
            AvailableSampler(
                value=stype.value,
                name=SCHEDULER_NAMES[stype],
                description=SCHEDULER_DESCRIPTIONS[stype],
            )
            for stype in SamplerType
        ]


# Singleton instance
model_manager = ModelManager()
