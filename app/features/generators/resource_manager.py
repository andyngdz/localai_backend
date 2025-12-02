"""Resource management and cleanup for image generation."""

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
		progress_callback.reset()

	def cleanup_after_generation(self) -> None:
		"""Clean up resources after generation completes or fails.

		Clears GPU cache and resets progress tracking state.
		"""
		# Final safety cleanup
		memory_manager.clear_cache()

		# Reset callback state
		progress_callback.reset()

	def handle_oom_error(self) -> None:
		"""Handle out-of-memory errors by clearing caches.

		Clears both GPU cache and tensor cache if available.
		Logs the error for monitoring.
		"""
		logger.error('Out of memory error during image generation - clearing all caches to recover')

		# Clear cache to recover from OOM
		memory_manager.clear_cache()

		# Clear tensor cache
		image_processor.clear_tensor_cache()


resource_manager = ResourceManager()
