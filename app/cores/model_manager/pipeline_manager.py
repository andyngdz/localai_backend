"""Pipeline instance and configuration management."""

import logging
from typing import Any, Optional

from app.cores.constants.samplers import DEFAULT_SAMPLE_SIZE
from app.cores.samplers import SCHEDULER_MAPPING, SamplerType

logger = logging.getLogger(__name__)


class PipelineManager:
	"""Manages the active diffusion pipeline and its configuration."""

	def __init__(self):
		self.pipe: Any = None
		self.model_id: Optional[str] = None

	def set_pipeline(self, pipe, model_id: str) -> None:
		"""Store pipeline and model ID.

		Args:
			pipe: Pipeline instance to store
			model_id: Model identifier
		"""
		self.pipe = pipe
		self.model_id = model_id
		logger.info(f'Pipeline set for model: {model_id}')

	def clear_pipeline(self) -> None:
		"""Clear pipeline and model ID."""
		self.pipe = None
		self.model_id = None
		logger.info('Pipeline cleared')

	def get_pipeline(self):
		"""Get current pipeline.

		Returns:
			Current pipeline instance or None if not loaded
		"""
		return self.pipe

	def get_model_id(self) -> Optional[str]:
		"""Get current model ID.

		Returns:
			Current model ID or None if not loaded
		"""
		return self.model_id

	def set_sampler(self, sampler: SamplerType) -> None:
		"""Dynamically set the sampler for the loaded pipeline.

		Args:
			sampler: Sampler type to set

		Raises:
			ValueError: If no model is loaded or sampler is unsupported
		"""
		if not self.pipe:
			raise ValueError('No model loaded')

		scheduler = SCHEDULER_MAPPING.get(sampler)
		if not scheduler:
			raise ValueError(f'Unsupported sampler type: {sampler.value}')

		config = self.pipe.scheduler.config
		kwargs = {}

		if sampler in [SamplerType.DPM_SOLVER_MULTISTEP_KARRAS, SamplerType.DPM_SOLVER_SDE_KARRAS]:
			kwargs['use_karras_sigmas'] = True

		new_scheduler = scheduler.from_config(config, **kwargs)
		self.pipe.scheduler = new_scheduler
		logger.info(f'Sampler set to: {sampler.value}')

	def get_sample_size(self) -> int:
		"""Get sample size from pipeline's UNet configuration.

		Returns:
			Sample size from UNet config or default value

		Raises:
			ValueError: If no model is loaded
		"""
		if not self.pipe:
			raise ValueError('No model loaded')

		unet_config = self.pipe.unet.config
		if hasattr(unet_config, 'sample_size'):
			sample_size: int = unet_config.sample_size
			return sample_size
		else:
			return DEFAULT_SAMPLE_SIZE


pipeline_manager = PipelineManager()
