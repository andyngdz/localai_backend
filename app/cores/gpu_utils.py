"""GPU utility functions for memory management."""

import gc
import time

import torch
from pydantic import BaseModel, Field

from app.services import logger_service

logger = logger_service.get_logger(__name__)


class CleanupMetrics(BaseModel):
	"""Metrics from GPU model cleanup operation."""

	time_ms: float = Field(description='Cleanup time in milliseconds')
	objects_collected: int = Field(description='Number of objects collected by GC')
	error: str | None = Field(default=None, description='Error message if cleanup failed')


def cleanup_gpu_model(model, name: str = 'model') -> CleanupMetrics:
	"""Clean up a GPU model and free VRAM.

	Args:
		model: The model/pipeline to clean up
		name: Name for logging purposes

	Returns:
		CleanupMetrics: Cleanup metrics with timing and GC stats
	"""
	if model is None:
		return CleanupMetrics(time_ms=0, objects_collected=0)

	start = time.time()

	try:
		del model
		collected = gc.collect()

		if torch.cuda.is_available():
			torch.cuda.empty_cache()

		elapsed_ms = (time.time() - start) * 1000
		logger.info(f'Cleaned up {name}: {elapsed_ms:.1f}ms, {collected} objects collected')

		return CleanupMetrics(time_ms=elapsed_ms, objects_collected=collected)

	except Exception as e:
		logger.warning(f'Error during {name} cleanup: {e}')
		return CleanupMetrics(time_ms=0, objects_collected=0, error=str(e))
