import asyncio
import gc
import logging
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Optional

import torch

from app.cores.constants.samplers import DEFAULT_SAMPLE_SIZE
from app.cores.model_loader import CancellationException, CancellationToken, model_loader
from app.cores.model_loader.schemas import ModelLoadCompletedResponse
from app.cores.samplers import (
	SCHEDULER_MAPPING,
	SamplerType,
)
from app.services import device_service
from app.socket import socket_service

logger = logging.getLogger(__name__)


class ModelState(Enum):
	"""Model loading states for tracking operations."""

	IDLE = 'idle'  # No model loaded
	LOADING = 'loading'  # Model load in progress
	LOADED = 'loaded'  # Model successfully loaded
	UNLOADING = 'unloading'  # Model unload in progress
	CANCELLING = 'cancelling'  # Load operation being cancelled
	ERROR = 'error'  # Previous operation failed


class ModelManager:
	"""
	Manages the active diffusion pipeline and handles background loading with
	proper concurrency control, cancellation support, and state management.
	"""

	def __init__(self):
		self.pipe = None
		self.id = None
		self.state: ModelState = ModelState.IDLE

		# Asyncio lock to prevent concurrent model operations
		self.lock = asyncio.Lock()

		# Track cancellation token for current load operation
		self.cancel_token: Optional[CancellationToken] = None

		# Track current loading task
		self.loading_task: Optional[asyncio.Task] = None

		# Single-threaded executor to serialize model loading
		# CRITICAL: max_workers=1 ensures only ONE load at a time
		self.executor = ThreadPoolExecutor(max_workers=1)

		logger.info('ModelManager instance initialized with concurrency controls.')

	def get_state(self) -> ModelState:
		"""Get current model state (thread-safe).

		Returns:
			Current ModelState
		"""
		return self.state

	def set_state(self, new_state: ModelState, reason: str = '') -> None:
		"""Set model state with logging.

		Args:
			new_state: New state to transition to
			reason: Optional reason for transition
		"""
		old_state = self.state
		self.state = new_state
		logger.info(f'Model state transition: {old_state.value} -> {new_state.value} {reason}')

	def release_resources(self):
		"""Clears the GPU cache with synchronous cleanup and verification."""

		logger.info(f'Starting resource release. Current state: {self.state.value}')

		# Store reference to pipe before deletion
		pipe_to_delete = self.pipe

		# Clear instance variables first
		self.pipe = None
		self.id = None

		if pipe_to_delete is not None:
			# Delete the pipeline object directly (no need to move to CPU first)
			del pipe_to_delete
			logger.info('Pipeline object deleted.')

		# Force garbage collection BEFORE clearing cache
		gc.collect()
		logger.info('Garbage collection completed (1st pass).')

		# Synchronize device operations and clear cache
		if device_service.is_available:
			if device_service.is_cuda:
				# Wait for all CUDA operations to complete
				torch.cuda.synchronize()
				logger.info('CUDA synchronized - all pending operations completed.')

				# Get memory stats before cleanup
				allocated_before = torch.cuda.memory_allocated() / (1024**3)
				reserved_before = torch.cuda.memory_reserved() / (1024**3)
				logger.info(f'GPU memory before cleanup: {allocated_before:.2f}GB allocated, {reserved_before:.2f}GB reserved')

				# Clear cache
				torch.cuda.empty_cache()

				# Force another GC pass after cache clear
				gc.collect()

				# Get memory stats after cleanup
				allocated_after = torch.cuda.memory_allocated() / (1024**3)
				reserved_after = torch.cuda.memory_reserved() / (1024**3)
				logger.info(f'GPU memory after cleanup: {allocated_after:.2f}GB allocated, {reserved_after:.2f}GB reserved')
				logger.info(
					f'GPU memory freed: {allocated_before - allocated_after:.2f}GB allocated, '
					f'{reserved_before - reserved_after:.2f}GB reserved'
				)

			elif device_service.is_mps:
				torch.mps.synchronize()
				logger.info('MPS synchronized - all pending operations completed.')
				torch.mps.empty_cache()
				gc.collect()
				logger.info('MPS cache cleared.')
		else:
			logger.warning('GPU acceleration is not available, cannot clear cache.')

		# Final GC pass
		gc.collect()
		logger.info('Final garbage collection completed (2nd pass).')

	def unload_internal(self):
		"""Internal unload method (no state checks, assumes already in correct state)."""
		if self.pipe is not None:
			logger.info(f'[unload_internal] Unloading model: {self.id}')
			self.release_resources()

	def load_model(self, id: str):
		"""
		Load a model synchronously into memory for inference.
		Should only be called when model is confirmed downloaded.

		NOTE: This is called from within the locked executor context,
		so we don't need additional locking here.

		Args:
			id: Model identifier to load

		Returns:
			Model configuration dictionary

		Raises:
			ValueError: If model loading fails
		"""

		logger.info(f'[load_model] Attempting to load model: {id}, current state: {self.state.value}')

		# Check if already loaded
		if self.id == id and self.pipe is not None and self.state == ModelState.LOADED:
			logger.info(f'Model {id} is already loaded and state is LOADED.')

			unet_config = self.pipe.unet.config
			if unet_config is not None:
				logger.info(f'UNet config: {unet_config}')

			config = dict(self.pipe.config)

			socket_service.model_load_completed(ModelLoadCompletedResponse(id=id))

			return config

		try:
			# Unload any existing model first
			if self.pipe is not None:
				logger.info(f'Unloading existing model {self.id} before loading {id}')
				self.unload_internal()

			# Load the new model with cancellation token
			logger.info(f'Starting model_loader for {id} with cancellation support')
			self.pipe = model_loader(id, self.cancel_token)

			logger.info(f'Model {id} loaded successfully.')

			self.id = id

			return dict(self.pipe.config)

		except CancellationException:
			logger.info(f'Model loading cancelled for {id}')
			self.unload_internal()
			raise

		except Exception as error:
			logger.error(f'Failed to load model {id}: {error}')
			self.unload_internal()
			raise

	async def load_model_async(self, id: str):
		"""
		Asynchronously load a model into memory for inference with cancellation support.

		This method automatically cancels any in-progress load before starting a new one,
		making it safe for React double-mount scenarios.

		Args:
			id: Model identifier to load

		Returns:
			Model configuration dictionary

		Raises:
			ValueError: If model loading fails or is in invalid state
			CancellationException: If loading is cancelled
		"""

		logger.info(f'[load_model_async] Request to load model: {id}, current state: {self.state.value}')

		# Before acquiring lock, check if we need to cancel a previous load
		if self.state == ModelState.LOADING:
			logger.info(f'Another load in progress, cancelling it for new load: {id}')

			# Set cancellation flag (thread-safe)
			if self.cancel_token:
				self.cancel_token.cancel()
				logger.info('Cancellation token flagged for previous load')

			# Wait for previous load to complete cancellation
			if self.loading_task and not self.loading_task.done():
				logger.info('Waiting for previous load to complete cancellation')
				try:
					await self.loading_task
				except (ValueError, CancellationException, asyncio.CancelledError):
					logger.info('Previous load cancelled successfully')
				except Exception as e:
					logger.warning(f'Previous load failed: {e}')

		# Acquire the lock to serialize model operations
		async with self.lock:
			logger.info(f'[load_model_async] Lock acquired for model: {id}')

			# Check if model is already loaded (fast path)
			if self.id == id and self.pipe is not None and self.state == ModelState.LOADED:
				logger.info(f'Model {id} is already loaded. Returning existing config.')
				return dict(self.pipe.config)

			# Check if we can transition to LOADING
			# Note: LOADING is allowed here because we auto-cancelled the previous load above
			if self.state not in [ModelState.IDLE, ModelState.ERROR, ModelState.LOADING]:
				error_msg = f'Cannot load model in state {self.state.value}. Current model: {self.id}'
				logger.error(error_msg)
				raise ValueError(error_msg)

			# Transition to LOADING state
			self.set_state(ModelState.LOADING, f'(id={id})')

			# Create new cancellation token for this load
			self.cancel_token = CancellationToken()

			# Track the current task for cancellation
			self.loading_task = asyncio.current_task()

		# Load model outside the lock to allow cancellation
		try:
			logger.info(f'[load_model_async] Submitting load_model to executor for {id}')

			# Execute the blocking load operation in thread pool
			loop = asyncio.get_event_loop()

			config = await loop.run_in_executor(
				self.executor,
				self.load_model,
				id,
			)

			# Acquire lock to update state
			async with self.lock:
				# Success - update state
				self.set_state(ModelState.LOADED, f'(id={id})')
				self.loading_task = None
				self.cancel_token = None

				logger.info(f'[load_model_async] Successfully loaded {id}')
				return config

		except CancellationException:
			# Model loader detected cancellation
			async with self.lock:
				self.set_state(ModelState.IDLE, '(cancelled)')
				self.loading_task = None
				self.cancel_token = None
			logger.info(f'Model loading cancelled for {id}')
			raise  # Let it propagate to API layer

		except Exception as error:
			# Other errors
			async with self.lock:
				self.set_state(ModelState.ERROR, f'(error: {error})')
				self.loading_task = None
				self.cancel_token = None
			logger.error(f'[load_model_async] Error loading model {id}: {error}')
			raise

	async def unload_model_async(self) -> None:
		"""
		Async unload with cancellation of in-progress loads.

		This method can cancel ongoing load operations before unloading.
		It's safe to call during React useEffect cleanup.
		"""

		logger.info(f'[unload_model_async] Unload requested, current state: {self.state.value}')

		# First, request cancellation if loading
		if self.state == ModelState.LOADING:
			logger.info('Cancelling in-progress model load')

			# Set cancellation flag (this is thread-safe)
			if self.cancel_token:
				self.cancel_token.cancel()
				logger.info('Cancellation token flagged')

			# Wait for the loading task to complete (it will detect cancellation and finish)
			if self.loading_task and not self.loading_task.done():
				logger.info('Waiting for load task to complete cancellation')
				try:
					await self.loading_task
				except (ValueError, CancellationException, asyncio.CancelledError):
					logger.info('Load task cancelled successfully')
				except Exception as e:
					logger.warning(f'Unexpected error during load cancellation: {e}')

			# After cancellation, check if load was successfully cancelled
			# If state is IDLE or ERROR, the cancelled load already cleaned up
			# Return early to avoid interfering with subsequent load operations
			if self.state in [ModelState.IDLE, ModelState.ERROR]:
				logger.info(f'Load successfully cancelled, state is {self.state.value}')
				return

		# Now acquire lock for actual unload
		async with self.lock:
			if self.state == ModelState.IDLE:
				logger.info('No model loaded, nothing to unload')
				return

			if self.state == ModelState.LOADED:
				self.set_state(ModelState.UNLOADING, f'(id={self.id})')

				try:
					# Unload current model
					self.unload_internal()
					self.set_state(ModelState.IDLE)
					logger.info('Model unloaded successfully')

				except Exception as e:
					self.set_state(ModelState.ERROR, f'(unload error: {e})')
					logger.error(f'Error unloading model: {e}')
					raise

			elif self.state in [ModelState.CANCELLING, ModelState.ERROR]:
				# Reset to IDLE
				self.unload_internal()
				self.set_state(ModelState.IDLE, '(reset from error/cancelling)')
				logger.info('Reset to IDLE state')

	def unload_model(self):
		"""Unloads the current model and frees VRAM (synchronous wrapper).

		This is kept for backwards compatibility with synchronous code.
		For async contexts, use unload_model_async() instead.
		"""

		logger.info(f'[unload_model] Sync unload requested, current state: {self.state.value}')

		try:
			if self.pipe is not None:
				self.set_state(ModelState.UNLOADING, f'(id={self.id})')
				self.unload_internal()
				self.set_state(ModelState.IDLE)
			else:
				logger.info('No model loaded, nothing to unload.')
		except Exception as error:
			logger.warning(f'Error during unload: {error}')
			self.set_state(ModelState.ERROR, f'(unload error: {error})')

	def set_sampler(self, sampler: SamplerType):
		"""Dynamically sets the sampler for the currently loaded pipeline.

		Args:
			sampler: Sampler type to set

		Raises:
			ValueError: If no model is loaded or state is not LOADED
		"""

		if not self.pipe or self.state != ModelState.LOADED:
			raise ValueError(f'No model loaded or model not in LOADED state. Current state: {self.state.value}')

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

	def get_sample_size(self):
		"""Returns the sample size of the model based on its configuration.

		Returns:
			Model sample size

		Raises:
			ValueError: If no model is loaded or state is not LOADED
		"""

		if not self.pipe or self.state != ModelState.LOADED:
			raise ValueError(f'No model loaded or model not in LOADED state. Current state: {self.state.value}')

		unet_config = self.pipe.unet.config

		if hasattr(unet_config, 'sample_size'):
			sample_size = unet_config.sample_size

			return sample_size
		else:
			return DEFAULT_SAMPLE_SIZE


model_manager = ModelManager()
