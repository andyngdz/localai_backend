"""Model loading orchestration with cancellation support."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from app.cores.model_loader import CancellationException, CancellationToken, model_loader
from app.cores.model_loader.schemas import ModelLoadCompletedResponse
from app.socket import socket_service

from .pipeline_manager import PipelineManager
from .resource_manager import ResourceManager
from .state_manager import ModelState, StateManager, StateTransitionReason

logger = logging.getLogger(__name__)


class LoaderService:
	"""Orchestrates model loading with state management and cancellation."""

	def __init__(
		self, state_manager: StateManager, resource_manager: ResourceManager, pipeline_manager: PipelineManager
	) -> None:
		self.state = state_manager
		self.resources = resource_manager
		self.pipeline = pipeline_manager

		self.lock = asyncio.Lock()
		self.cancel_token: Optional[CancellationToken] = None
		self.loading_task: Optional[asyncio.Task] = None

		self.executor = ThreadPoolExecutor(max_workers=1)

		logger.info('LoaderService initialized with concurrency controls')

	def shutdown(self) -> None:
		"""Shutdown the executor thread pool."""
		self.executor.shutdown(wait=True)
		logger.info('LoaderService executor shut down')

	async def load_model_async(self, id: str) -> dict[str, object]:
		"""Load model asynchronously with automatic cancellation of previous loads.

		Args:
			id: Model identifier to load

		Returns:
			Model configuration dictionary

		Raises:
			ValueError: If model loading fails or is in invalid state
			CancellationException: If loading is cancelled
		"""
		async with self.lock:
			current_state = self.state.current_state
			logger.info(f'[load_model_async] Request to load: {id}, current state: {current_state.value}')

			if current_state == ModelState.LOADING:
				logger.info(f'Another load in progress, need to cancel for new load: {id}')

		if current_state == ModelState.LOADING:
			await self.cancel_current_load()
			async with self.lock:
				current_state = self.state.current_state

		async with self.lock:
			if self.pipeline.model_id == id and self.pipeline.pipe is not None and current_state == ModelState.LOADED:
				logger.info(f'Model {id} already loaded, returning config')
				return dict(self.pipeline.pipe.config)

			if not self.state.can_transition_to(ModelState.LOADING):
				error_msg = f'Cannot load model in state {current_state.value}'
				logger.error(error_msg)
				raise ValueError(error_msg)

			self.state.set_state(ModelState.LOADING, StateTransitionReason.LOAD_REQUESTED)
			self.cancel_token = CancellationToken()
			self.loading_task = asyncio.current_task()

		try:
			logger.info(f'[load_model_async] Executing load for {id}')
			config = await self.execute_load_in_background(id)

			async with self.lock:
				self.state.set_state(ModelState.LOADED, StateTransitionReason.LOAD_COMPLETED)
				self.loading_task = None
				self.cancel_token = None
				logger.info(f'[load_model_async] Successfully loaded {id}')

			return config

		except CancellationException:
			async with self.lock:
				self.state.set_state(ModelState.IDLE, StateTransitionReason.LOAD_CANCELLED)
				self.loading_task = None
				self.cancel_token = None
			logger.info(f'Model loading cancelled for {id}')
			raise

		except Exception as error:
			async with self.lock:
				self.state.set_state(ModelState.ERROR, StateTransitionReason.LOAD_FAILED)
				self.loading_task = None
				self.cancel_token = None
			logger.error(f'[load_model_async] Error loading {id}: {error}')
			raise

	async def unload_model_async(self) -> None:
		"""Unload model asynchronously with cancellation support.

		This method can cancel ongoing load operations before unloading.
		It's safe to call during React useEffect cleanup.
		"""
		async with self.lock:
			current_state = self.state.current_state
			logger.info(f'[unload_model_async] Request to unload, current state: {current_state.value}')

			if current_state == ModelState.LOADING:
				logger.info('Cancelling in-progress load before unload')

		if current_state == ModelState.LOADING:
			await self.cancel_current_load()
			async with self.lock:
				current_state = self.state.current_state

				if current_state in {ModelState.IDLE, ModelState.ERROR}:
					logger.info(f'Load cancelled successfully, state: {current_state.value}')
					return

		async with self.lock:
			if current_state == ModelState.IDLE:
				logger.info('No model loaded, nothing to unload')
				return

			if current_state == ModelState.LOADED:
				self.state.set_state(ModelState.UNLOADING, StateTransitionReason.UNLOAD_REQUESTED)

				try:
					self.unload_model_sync()
					self.state.set_state(ModelState.IDLE, StateTransitionReason.UNLOAD_COMPLETED)
					logger.info('Model unloaded successfully')
				except Exception as e:
					self.state.set_state(ModelState.ERROR, StateTransitionReason.UNLOAD_FAILED)
					logger.error(f'Error unloading model: {e}')
					raise

			elif current_state == ModelState.ERROR:
				self.unload_model_sync()
				self.state.set_state(ModelState.IDLE, StateTransitionReason.RESET_FROM_ERROR)
				logger.info('Reset to IDLE state')

	async def cancel_current_load(self) -> None:
		"""Cancel any in-progress model loading operation.

		This method is safe to call even if no load is in progress.
		"""
		if self.cancel_token:
			self.cancel_token.cancel()
			logger.info('Cancellation token flagged')

		if self.loading_task and not self.loading_task.done():
			logger.info('Waiting for load task to complete cancellation')
			try:
				await self.loading_task
			except (ValueError, CancellationException, asyncio.CancelledError):
				logger.info('Load task cancelled successfully')
			except Exception as e:
				logger.warning(f'Unexpected error during load cancellation: {e}')

	async def execute_load_in_background(self, id: str) -> dict:
		"""Execute model loading in background thread pool.

		Args:
			id: Model identifier to load

		Returns:
			Model configuration dictionary
		"""
		loop = asyncio.get_event_loop()
		return await loop.run_in_executor(self.executor, self.load_model_sync, id)

	def load_model_sync(self, id: str) -> dict[str, object]:
		"""Synchronous model loading (called from executor thread).

		Args:
			id: Model identifier to load

		Returns:
			Model configuration dictionary

		Raises:
			CancellationException: If loading is cancelled via cancel_token
		"""
		logger.info(f'[load_model_sync] Loading model: {id}')

		if self.pipeline.pipe is not None:
			logger.info(f'Unloading existing model {self.pipeline.model_id} before loading {id}')
			self.unload_model_sync()

		logger.info(f'Starting model_loader for {id}')
		pipe = model_loader(id, self.cancel_token)

		logger.info(f'Model {id} loaded successfully')
		self.pipeline.set_pipeline(pipe, id)

		socket_service.model_load_completed(ModelLoadCompletedResponse(id=id))

		return dict(pipe.config)

	def unload_model_sync(self) -> None:
		"""Synchronous model unloading.

		Cleans up pipeline resources and clears pipeline reference.
		Safe to call even if no model is loaded.
		"""
		if self.pipeline.pipe is not None and self.pipeline.model_id is not None:
			logger.info(f'[unload_model_sync] Unloading model: {self.pipeline.model_id}')
			self.resources.cleanup_pipeline(self.pipeline.pipe, self.pipeline.model_id)
			self.pipeline.clear_pipeline()
