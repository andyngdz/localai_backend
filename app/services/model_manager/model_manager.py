import logging
import threading
from typing import Any, Dict, Optional

import torch
from diffusers.pipelines import AutoPipelineForText2Image

from config import BASE_CACHE_DIR

from .schedulers import SCHEDULER_DESCRIPTIONS, SCHEDULER_NAMES, SamplerType
from .schemas import AvailableSampler

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages the active diffusion pipeline, ensuring thread-safe loading and unloading.
    """

    instance: Optional['ModelManager'] = None
    lock: threading.Lock = threading.Lock()

    def __new__(cls):
        if cls.instance is None:
            with cls.lock:
                if cls.instance is None:
                    cls.instance = super(ModelManager, cls).__new__(cls)
                    cls.instance.pipe = None
                    cls.instance.current_id = None
                    logger.info('ModelManager instance created.')
        return cls.instance

    def load_model(self, id: str) -> Dict[str, Any]:
        """
        Loads a diffusion pipeline, handles existing models, and clears VRAM.
        This method is blocking and thread-safe.
        """

        with self.lock:
            logger.info(f'Attempting to load model: {id}')

            if self.current_id == id and self.pipe is not None:
                logger.info(f'Model {id} is already loaded. Skipping load operation.')
                return dict(self.pipe.config)

            if self.pipe is not None:
                logger.info(f'Unloading existing model: {self.current_id}')
                try:
                    del self.pipe
                    self.pipe = None
                    self.current_id = None

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info('CUDA cache emptied.')

                except Exception as e:
                    logger.warning(
                        f'Error during existing model unload/cache clear: {e}'
                    )

            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                logger.info(f'Loading model {id} to device: {device}')

                self.pipe = AutoPipelineForText2Image.from_pretrained(
                    id,
                    cache_dir=BASE_CACHE_DIR,
                    torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                    use_safetensors=True,
                )

                self.pipe.to(device)

                if device == 'cuda':
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

        with self.lock:
            if self.pipe is not None:
                logger.info(f'Manually unloading model: {self.current_id}')

                try:
                    del self.pipe

                    self.pipe = None
                    self.current_id = None

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info('CUDA cache emptied after manual unload.')
                except Exception as e:
                    logger.warning(f'Error during manual unload/cache clear: {e}')
            else:
                logger.info('No model currently loaded to unload.')

    def get_available_samplers(self) -> list[AvailableSampler]:
        """Returns a list of available samplers with their names and descriptions."""

        return [
            AvailableSampler(
                value=samplerType.value,
                name=SCHEDULER_NAMES[samplerType],
                description=SCHEDULER_DESCRIPTIONS[samplerType],
            )
            for samplerType in SamplerType
        ]


model_manager = ModelManager()
