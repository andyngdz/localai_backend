"""Model manager facade coordinating specialized managers."""

import logging
from typing import Optional

from app.cores.samplers import SamplerType

from .loader_service import LoaderService
from .pipeline_manager import PipelineManager
from .resource_manager import ResourceManager
from .state_manager import ModelState, StateManager, StateTransitionReason

logger = logging.getLogger(__name__)


class ModelManager:
	"""
	Facade for model loading, state management, and resource cleanup.

	Internally delegates to specialized managers:
	- StateManager: State machine transitions
	- ResourceManager: GPU/MPS memory cleanup
	- PipelineManager: Pipeline storage and configuration
	- LoaderService: Async loading orchestration
	"""

	def __init__(self) -> None:
		self.state_manager: StateManager = StateManager()
		self.resource_manager: ResourceManager = ResourceManager()
		self.pipeline_manager: PipelineManager = PipelineManager()
		self.loader_service: LoaderService = LoaderService(
			state_manager=self.state_manager, resource_manager=self.resource_manager, pipeline_manager=self.pipeline_manager
		)
		logger.info('ModelManager initialized with modular architecture')

	async def load_model_async(self, id: str) -> dict[str, object]:
		"""Load model asynchronously with cancellation support.

		Args:
			id: Model identifier to load

		Returns:
			Model configuration dictionary

		Raises:
			ValueError: If model loading fails or is in invalid state
			CancellationException: If loading is cancelled
		"""
		return await self.loader_service.load_model_async(id)

	async def unload_model_async(self) -> None:
		"""Unload model asynchronously.

		This method can cancel ongoing load operations before unloading.
		"""
		await self.loader_service.unload_model_async()

	@property
	def current_state(self) -> ModelState:
		"""Get current model state.

		Returns:
			Current ModelState
		"""
		return self.state_manager.get_state()

	def get_state(self) -> ModelState:
		"""Get current model state (deprecated: use current_state property).

		Returns:
			Current ModelState
		"""
		return self.current_state

	def set_state(self, new_state: ModelState, reason: StateTransitionReason) -> None:
		"""Set model state with logging.

		Args:
			new_state: New state to transition to
			reason: Predefined reason for transition
		"""
		self.state_manager.set_state(new_state, reason)

	def set_sampler(self, sampler: SamplerType) -> None:
		"""Set sampler for loaded pipeline.

		Args:
			sampler: Sampler type to set

		Raises:
			ValueError: If no model is loaded or sampler is unsupported
		"""
		self.pipeline_manager.set_sampler(sampler)

	@property
	def sample_size(self) -> int:
		"""Get sample size from pipeline.

		Returns:
			Sample size from UNet config or default value

		Raises:
			ValueError: If no model is loaded
		"""
		return self.pipeline_manager.get_sample_size()

	def get_sample_size(self) -> int:
		"""Get sample size from pipeline (deprecated: use sample_size property).

		Returns:
			Sample size from UNet config or default value

		Raises:
			ValueError: If no model is loaded
		"""
		return self.sample_size

	@property
	def pipe(self):
		"""Get current pipeline (backward compatibility).

		Returns:
			Current pipeline instance or None if not loaded
		"""
		return self.pipeline_manager.get_pipeline()

	@pipe.setter
	def pipe(self, value):
		"""Set pipeline directly (for img2img conversion).

		Args:
			value: Pipeline instance to set

		Raises:
			ValueError: If no model_id exists (must load model first)
		"""
		model_id = self.pipeline_manager.get_model_id()
		if model_id is None:
			raise ValueError('Cannot set pipeline without loading a model first')
		self.pipeline_manager.set_pipeline(value, model_id)

	@property
	def id(self) -> Optional[str]:
		"""Get current model ID (backward compatibility).

		Returns:
			Current model ID or None if not loaded
		"""
		return self.pipeline_manager.get_model_id()

	@id.setter
	def id(self, value: Optional[str]):
		"""Set model ID directly (for backward compatibility).

		Args:
			value: Model ID to set or None to clear
		"""
		self.pipeline_manager.model_id = value

	@property
	def lock(self):
		"""Expose lock for external synchronization (advanced use).

		Returns:
			AsyncIO lock instance
		"""
		return self.loader_service.lock


# Singleton instance
model_manager = ModelManager()
