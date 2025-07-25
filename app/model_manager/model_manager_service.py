import asyncio
import gc
import logging
from concurrent.futures import ThreadPoolExecutor

import torch

from app.constants import (
	DEFAULT_SAMPLE_SIZE,
	SCHEDULER_MAPPING,
	SamplerType,
)
from app.model_loader import model_loader
from app.services.device import device_service

logger = logging.getLogger(__name__)


class ModelManagerService:
	"""
	Manages the active diffusion pipeline and handles background loading with
	cancellation.
	"""

	def __init__(self):
		self.pipe = None
		self.id = None
		self.executor = ThreadPoolExecutor()

		logger.info('ModelManager instance initialized.')

	def release_resources(self):
		"""Clears the CUDA cache if available."""

		del self.pipe
		self.pipe = None
		self.id = None

		if device_service.is_available:
			torch.cuda.empty_cache()
			logger.info('CUDA cache cleared.')
		else:
			logger.warning('CUDA is not available, cannot clear cache.')

		gc.collect()
		logging.info('Forcing garbage collection to free memory.')

	def load_model(self, id: str):
		"""
		Load a model synchronously into memory for inference.
		Should only be called when model is confirmed downloaded.
		"""

		logger.info(f'Attempting to load model: {id}')

		if self.id == id and self.pipe is not None:
			logger.info(f'Model {id} is already loaded.')

			unet_config = self.pipe.unet.config
			if unet_config is not None:
				logger.info(f'UNet config: {unet_config}')

			return dict(self.pipe.config)

		try:
			self.unload_model()
			self.pipe = model_loader(id)

			logger.info(f'Model {id} loaded successfully.')

			self.id = id

			return dict(self.pipe.config)

		except Exception as error:
			self.unload_model()
			logger.error(f'Failed to load model {id}: {error}')
			raise

	async def load_model_async(self, id: str):
		"""
		Asynchronously load a model into memory for inference.
		This method is intended to be run in a separate thread.
		"""

		logger.info(f'Asynchronously loading model: {id}')

		try:
			loop = asyncio.get_event_loop()

			config = await loop.run_in_executor(
				self.executor,
				model_manager_service.load_model,
				id,
			)

			return config
		except Exception as error:
			logger.error(f'Error loading model {id} asynchronously: {error}')

	def unload_model(self):
		"""Unloads the current model and frees VRAM."""

		try:
			if self.pipe is not None:
				logger.info(f'Unloading model: {self.id}')
				
				self.release_resources()
		except Exception as error:
			logger.warning(f'Error during unload: {error}')

	def set_sampler(self, sampler: SamplerType):
		"""Dynamically sets the sampler for the currently loaded pipeline."""

		if not self.pipe:
			raise ValueError('No model loaded. Cannot set sampler.')

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
		"""Returns the sample size of the model based on its configuration."""

		if not self.pipe:
			raise ValueError('No model loaded. Cannot get sample size.')

		unet_config = self.pipe.unet.config

		if hasattr(unet_config, 'sample_size'):
			sample_size = unet_config.sample_size

			return sample_size
		else:
			return DEFAULT_SAMPLE_SIZE


model_manager_service = ModelManagerService()
