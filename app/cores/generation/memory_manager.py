"""Memory management utilities for GPU/CPU resource optimization."""

import torch

from app.services import device_service, logger_service

logger = logger_service.get_logger(__name__, category='Generate')


class MemoryManager:
	"""Manages GPU memory and provides batch size recommendations."""

	def clear_cache(self) -> None:
		"""Clear GPU cache if available.

		This helps prevent out-of-memory errors by freeing unused memory.
		"""
		if device_service.is_cuda:
			torch.cuda.empty_cache()
			logger.info('Cleared CUDA cache')
		elif device_service.is_mps:
			torch.mps.empty_cache()
			logger.info('Cleared MPS cache')

	def validate_batch_size(self, number_of_images: int, width: int, height: int) -> None:
		"""Validate batch size and log warnings if it may cause OOM.

		Args:
			number_of_images: Number of images to generate in one batch.
			width: Image width in pixels.
			height: Image height in pixels.
		"""
		recommended_batch = device_service.get_recommended_batch_size()
		if number_of_images > recommended_batch:
			logger.warning(
				f'Generating {number_of_images} images at {width}x{height} may cause OOM errors. '
				f'Recommended: {recommended_batch}'
			)


memory_manager = MemoryManager()
