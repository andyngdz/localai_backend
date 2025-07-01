import logging
import threading
from typing import Any, Dict, Optional

import torch
from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages the active diffusion pipeline, ensuring thread-safe loading and unloading.
    """

    _instance: Optional['ModelManager'] = None
    _lock: threading.Lock = threading.Lock()  # For thread-safe operations on _pipe

    def __new__(cls):
        # Implement a basic singleton pattern
        if cls._instance is None:
            with cls._lock:  # Ensure only one instance is created
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
                    cls._instance._pipe = None
                    cls._instance._current_model_id = None
                    logger.info('ModelManager instance created.')
        return cls._instance

    @property
    def current_model_id(self) -> Optional[str]:
        """Returns the ID of the currently loaded model."""
        return self._current_model_id

    @property
    def pipe(self) -> Optional[AutoPipelineForText2Image]:
        """Returns the currently active pipeline."""
        return self._pipe

    def load_model(self, model_id: str, model_dir: str) -> Dict[str, Any]:
        """
        Loads a diffusion pipeline, handles existing models, and clears VRAM.
        This method is blocking and thread-safe.
        """
        with self._lock:  # Acquire lock for the entire load/unload process
            logger.info(f'Attempting to load model: {model_id}')

            if self._current_model_id == model_id and self._pipe is not None:
                logger.info(
                    f'Model {model_id} is already loaded. Skipping load operation.'
                )
                return dict(self._pipe.config)  # Return existing config

            # Unload any existing model first
            if self._pipe is not None:
                logger.info(f'Unloading existing model: {self._current_model_id}')
                try:
                    del self._pipe
                    self._pipe = None
                    self._current_model_id = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info('CUDA cache emptied.')
                except Exception as e:
                    logger.warning(
                        f'Error during existing model unload/cache clear: {e}'
                    )

            # Load the new model
            try:
                # Determine device
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                logger.info(f'Loading model {model_id} to device: {device}')

                # Model loading is heavy, log before and after
                self._pipe = AutoPipelineForText2Image.from_pretrained(
                    model_dir,
                    torch_dtype=torch.float16
                    if device == 'cuda'
                    else torch.float32,  # Use float32 for CPU
                    local_files_only=True,  # Assume model_dir means it's already downloaded locally
                )
                self._pipe.to(device)

                if device == 'cuda':
                    self._pipe.enable_model_cpu_offload()  # Moves parts to CPU when not used
                    self._pipe.enable_attention_slicing()  # Saves VRAM during attention computation

                self._current_model_id = model_id
                logger.info(f'Model {model_id} loaded successfully.')
                return dict(self._pipe.config)

            except Exception as e:
                self._pipe = None  # Ensure pipe is cleared on failure
                self._current_model_id = None
                logger.error(f'Failed to load model {model_id}: {e}', exc_info=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Try to clear cache even on failure
                raise  # Re-raise to be caught by the endpoint's error handler

    def unload_model(self):
        """Unloads the current model and frees VRAM."""
        with self._lock:
            if self._pipe is not None:
                logger.info(f'Manually unloading model: {self._current_model_id}')
                try:
                    del self._pipe
                    self._pipe = None
                    self._current_model_id = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info('CUDA cache emptied after manual unload.')
                except Exception as e:
                    logger.warning(f'Error during manual unload/cache clear: {e}')
            else:
                logger.info('No model currently loaded to unload.')


# Get a singleton instance of the ModelManager
model_manager = ModelManager()
