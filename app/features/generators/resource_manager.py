"""Resource management and cleanup for image generation."""

from typing import Callable, Optional

from app.cores.generation import image_processor, memory_manager, progress_callback
from app.services import logger_service

logger = logger_service.get_logger(__name__, category='Generate')


class ResourceManager:
	"""Manages resource cleanup and cache clearing."""

	def prepare_for_generation(self) -> None:
		"""Prepare resources before generation starts.

		Clears caches and resets progress tracking.
		"""
		# Clear CUDA cache before generation to maximize available memory
		memory_manager.clear_cache()

		# Reset progress callback state for new generation
		if hasattr(progress_callback, 'reset'):
			progress_callback.reset()

	def cleanup_after_generation(self, loras_loaded: bool, unload_loras_fn: Optional[Callable[[], None]] = None) -> None:
		"""Clean up resources after generation completes or fails.

		Args:
			loras_loaded: Whether LoRAs were loaded during generation
			unload_loras_fn: Function to call for unloading LoRAs (optional)
		"""
		# Unload LoRAs if they were loaded
		if loras_loaded and unload_loras_fn:
			unload_loras_fn()

		# Final safety cleanup
		memory_manager.clear_cache()

		# Reset callback state
		if hasattr(progress_callback, 'reset'):
			progress_callback.reset()

	def handle_oom_error(self) -> None:
		"""Handle out-of-memory errors by clearing caches.

		Clears both GPU cache and tensor cache if available.
		"""
		logger.error('Out of memory error - clearing caches')

		# Clear cache to recover from OOM
		memory_manager.clear_cache()

		# Clear tensor cache if available
		if hasattr(image_processor, 'clear_tensor_cache'):
			image_processor.clear_tensor_cache()


resource_manager = ResourceManager()
