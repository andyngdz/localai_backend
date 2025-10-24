"""Comprehensive tests for PipelineManager class.

This test suite covers:
1. Pipeline storage and retrieval (set_pipeline, get_pipeline, clear_pipeline)
2. Model ID management (get_model_id)
3. Sampler configuration (set_sampler) including Karras variants
4. Sample size retrieval (get_sample_size)
5. Error cases (no pipe loaded, unsupported sampler)
"""

from unittest.mock import MagicMock, patch

import pytest

from app.cores.model_manager.pipeline_manager import PipelineManager
from app.cores.samplers import SamplerType


class TestPipelineStorage:
	"""Test pipeline storage and retrieval methods."""

	def setup_method(self):
		"""Create fresh PipelineManager for each test."""
		self.pipeline_manager = PipelineManager()

	def test_initial_pipeline_is_none(self):
		"""Test that pipeline is None on initialization."""
		assert self.pipeline_manager.get_pipeline() is None
		assert self.pipeline_manager.get_model_id() is None

	def test_set_pipeline_stores_pipe_and_model_id(self):
		"""Test set_pipeline stores both pipe and model_id."""
		mock_pipe = MagicMock()

		self.pipeline_manager.set_pipeline(mock_pipe, 'test/model')

		assert self.pipeline_manager.get_pipeline() == mock_pipe
		assert self.pipeline_manager.get_model_id() == 'test/model'

	def test_set_pipeline_logs_model_id(self, caplog):
		"""Test set_pipeline logs the model_id."""
		import logging

		caplog.set_level(logging.INFO)
		mock_pipe = MagicMock()

		self.pipeline_manager.set_pipeline(mock_pipe, 'my/model')

		assert 'Pipeline set for model: my/model' in caplog.text

	def test_clear_pipeline_resets_pipe_and_model_id(self):
		"""Test clear_pipeline sets pipe and model_id to None."""
		mock_pipe = MagicMock()
		self.pipeline_manager.set_pipeline(mock_pipe, 'test/model')

		self.pipeline_manager.clear_pipeline()

		assert self.pipeline_manager.get_pipeline() is None
		assert self.pipeline_manager.get_model_id() is None

	def test_clear_pipeline_logs_operation(self, caplog):
		"""Test clear_pipeline logs the operation."""
		import logging

		caplog.set_level(logging.INFO)
		mock_pipe = MagicMock()
		self.pipeline_manager.set_pipeline(mock_pipe, 'test/model')

		self.pipeline_manager.clear_pipeline()

		assert 'Pipeline cleared' in caplog.text

	def test_get_pipeline_returns_current_pipe(self):
		"""Test get_pipeline returns the current pipeline."""
		mock_pipe = MagicMock()
		self.pipeline_manager.pipe = mock_pipe

		result = self.pipeline_manager.get_pipeline()

		assert result == mock_pipe

	def test_get_model_id_returns_current_id(self):
		"""Test get_model_id returns the current model ID."""
		self.pipeline_manager.model_id = 'my/awesome/model'

		result = self.pipeline_manager.get_model_id()

		assert result == 'my/awesome/model'


class TestSetSampler:
	"""Test set_sampler() method with various sampler types."""

	def setup_method(self):
		"""Create fresh PipelineManager for each test."""
		self.pipeline_manager = PipelineManager()

	def test_set_sampler_raises_when_no_pipe_loaded(self):
		"""Test set_sampler raises ValueError when no pipe is loaded."""
		self.pipeline_manager.pipe = None

		with pytest.raises(ValueError, match='No model loaded'):
			self.pipeline_manager.set_sampler(SamplerType.EULER)

	def test_set_sampler_with_euler(self):
		"""Test set_sampler with EULER sampler."""
		# Setup
		mock_scheduler_config = {'key': 'value'}
		mock_scheduler = MagicMock()
		mock_scheduler.config = mock_scheduler_config

		mock_pipe = MagicMock()
		mock_pipe.scheduler = mock_scheduler

		self.pipeline_manager.pipe = mock_pipe

		# Mock SCHEDULER_MAPPING
		mock_new_scheduler_class = MagicMock()
		mock_new_scheduler_instance = MagicMock()
		mock_new_scheduler_class.from_config.return_value = mock_new_scheduler_instance

		with patch(
			'app.cores.model_manager.pipeline_manager.SCHEDULER_MAPPING', {SamplerType.EULER: mock_new_scheduler_class}
		):
			# Execute
			self.pipeline_manager.set_sampler(SamplerType.EULER)

			# Verify
			mock_new_scheduler_class.from_config.assert_called_once_with(mock_scheduler_config)
			assert self.pipeline_manager.pipe.scheduler == mock_new_scheduler_instance

	def test_set_sampler_with_karras_dpm_solver_multistep(self):
		"""Test set_sampler with DPM_SOLVER_MULTISTEP_KARRAS (Karras variant)."""
		# Setup
		mock_scheduler_config = {'key': 'value'}
		mock_scheduler = MagicMock()
		mock_scheduler.config = mock_scheduler_config

		mock_pipe = MagicMock()
		mock_pipe.scheduler = mock_scheduler

		self.pipeline_manager.pipe = mock_pipe

		# Mock SCHEDULER_MAPPING
		mock_new_scheduler_class = MagicMock()
		mock_new_scheduler_instance = MagicMock()
		mock_new_scheduler_class.from_config.return_value = mock_new_scheduler_instance

		with patch(
			'app.cores.model_manager.pipeline_manager.SCHEDULER_MAPPING',
			{SamplerType.DPM_SOLVER_MULTISTEP_KARRAS: mock_new_scheduler_class},
		):
			# Execute
			self.pipeline_manager.set_sampler(SamplerType.DPM_SOLVER_MULTISTEP_KARRAS)

			# Verify use_karras_sigmas=True was passed
			mock_new_scheduler_class.from_config.assert_called_once_with(mock_scheduler_config, use_karras_sigmas=True)
			assert self.pipeline_manager.pipe.scheduler == mock_new_scheduler_instance

	def test_set_sampler_with_karras_dpm_solver_sde(self):
		"""Test set_sampler with DPM_SOLVER_SDE_KARRAS (Karras variant)."""
		# Setup
		mock_scheduler_config = {'key': 'value'}
		mock_scheduler = MagicMock()
		mock_scheduler.config = mock_scheduler_config

		mock_pipe = MagicMock()
		mock_pipe.scheduler = mock_scheduler

		self.pipeline_manager.pipe = mock_pipe

		# Mock SCHEDULER_MAPPING
		mock_new_scheduler_class = MagicMock()
		mock_new_scheduler_instance = MagicMock()
		mock_new_scheduler_class.from_config.return_value = mock_new_scheduler_instance

		with patch(
			'app.cores.model_manager.pipeline_manager.SCHEDULER_MAPPING',
			{SamplerType.DPM_SOLVER_SDE_KARRAS: mock_new_scheduler_class},
		):
			# Execute
			self.pipeline_manager.set_sampler(SamplerType.DPM_SOLVER_SDE_KARRAS)

			# Verify use_karras_sigmas=True was passed
			mock_new_scheduler_class.from_config.assert_called_once_with(mock_scheduler_config, use_karras_sigmas=True)

	def test_set_sampler_raises_for_unsupported_sampler(self):
		"""Test set_sampler raises ValueError for unsupported sampler."""
		# Setup
		mock_pipe = MagicMock()
		mock_pipe.scheduler.config = {}
		self.pipeline_manager.pipe = mock_pipe

		# Mock SCHEDULER_MAPPING to be empty (no samplers)
		with patch('app.cores.model_manager.pipeline_manager.SCHEDULER_MAPPING', {}):
			with pytest.raises(ValueError, match='Unsupported sampler type'):
				self.pipeline_manager.set_sampler(SamplerType.EULER)

	def test_set_sampler_logs_sampler_change(self, caplog):
		"""Test set_sampler logs the sampler change."""
		import logging

		caplog.set_level(logging.INFO)

		# Setup
		mock_scheduler_config = {'key': 'value'}
		mock_scheduler = MagicMock()
		mock_scheduler.config = mock_scheduler_config

		mock_pipe = MagicMock()
		mock_pipe.scheduler = mock_scheduler

		self.pipeline_manager.pipe = mock_pipe

		# Mock SCHEDULER_MAPPING
		mock_new_scheduler_class = MagicMock()
		mock_new_scheduler_instance = MagicMock()
		mock_new_scheduler_class.from_config.return_value = mock_new_scheduler_instance

		with patch(
			'app.cores.model_manager.pipeline_manager.SCHEDULER_MAPPING', {SamplerType.EULER: mock_new_scheduler_class}
		):
			# Execute
			self.pipeline_manager.set_sampler(SamplerType.EULER)

			# Verify logging (SamplerType.EULER.value is uppercase "EULER")
			assert 'Sampler set to: EULER' in caplog.text


class TestGetSampleSize:
	"""Test get_sample_size() method."""

	def setup_method(self):
		"""Create fresh PipelineManager for each test."""
		self.pipeline_manager = PipelineManager()

	def test_get_sample_size_raises_when_no_pipe_loaded(self):
		"""Test get_sample_size raises ValueError when no pipe is loaded."""
		self.pipeline_manager.pipe = None

		with pytest.raises(ValueError, match='No model loaded'):
			self.pipeline_manager.get_sample_size()

	def test_get_sample_size_returns_unet_sample_size(self):
		"""Test get_sample_size returns sample_size from UNet config."""
		# Setup
		mock_unet_config = MagicMock()
		mock_unet_config.sample_size = 64

		mock_pipe = MagicMock()
		mock_pipe.unet.config = mock_unet_config

		self.pipeline_manager.pipe = mock_pipe

		# Execute
		result = self.pipeline_manager.get_sample_size()

		# Verify
		assert result == 64

	def test_get_sample_size_returns_default_when_no_sample_size_attribute(self):
		"""Test get_sample_size returns DEFAULT_SAMPLE_SIZE when sample_size not in config."""
		from app.cores.constants.samplers import DEFAULT_SAMPLE_SIZE

		# Setup - config without sample_size attribute
		mock_unet_config = MagicMock(spec_set=['other_attr'])

		mock_pipe = MagicMock()
		mock_pipe.unet.config = mock_unet_config

		self.pipeline_manager.pipe = mock_pipe

		# Execute
		result = self.pipeline_manager.get_sample_size()

		# Verify
		assert result == DEFAULT_SAMPLE_SIZE

	def test_get_sample_size_with_different_sample_sizes(self):
		"""Test get_sample_size with various sample size values."""
		# Test with sample_size=32
		mock_unet_config_32 = MagicMock()
		mock_unet_config_32.sample_size = 32

		mock_pipe = MagicMock()
		mock_pipe.unet.config = mock_unet_config_32

		self.pipeline_manager.pipe = mock_pipe

		assert self.pipeline_manager.get_sample_size() == 32

		# Test with sample_size=128
		mock_unet_config_128 = MagicMock()
		mock_unet_config_128.sample_size = 128

		mock_pipe.unet.config = mock_unet_config_128

		assert self.pipeline_manager.get_sample_size() == 128


class TestPipelineManagerEdgeCases:
	"""Test edge cases and state consistency."""

	def setup_method(self):
		"""Create fresh PipelineManager for each test."""
		self.pipeline_manager = PipelineManager()

	def test_set_pipeline_can_overwrite_existing_pipeline(self):
		"""Test set_pipeline can replace an existing pipeline."""
		old_pipe = MagicMock()
		new_pipe = MagicMock()

		self.pipeline_manager.set_pipeline(old_pipe, 'old/model')
		self.pipeline_manager.set_pipeline(new_pipe, 'new/model')

		assert self.pipeline_manager.get_pipeline() == new_pipe
		assert self.pipeline_manager.get_model_id() == 'new/model'

	def test_clear_pipeline_is_idempotent(self):
		"""Test clear_pipeline can be called multiple times safely."""
		mock_pipe = MagicMock()
		self.pipeline_manager.set_pipeline(mock_pipe, 'test/model')

		self.pipeline_manager.clear_pipeline()
		self.pipeline_manager.clear_pipeline()  # Call again

		assert self.pipeline_manager.get_pipeline() is None
		assert self.pipeline_manager.get_model_id() is None
