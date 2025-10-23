"""Comprehensive tests for ModelManager class.

This test suite covers:
1. State management (get_state, set_state)
2. release_resources() - GPU/MPS cleanup
3. load_model() - synchronous model loading
4. unload_model() - synchronous model unloading
5. set_sampler() - dynamic sampler switching
6. get_sample_size() - model dimension retrieval
"""

from unittest.mock import MagicMock, patch

import pytest

from app.cores.model_manager import ModelState, model_manager
from app.cores.samplers import SamplerType


class TestStateManagement:
	"""Test get_state() and set_state() methods."""

	def setup_method(self):
		"""Reset model_manager state before each test."""
		model_manager.pipe = None
		model_manager.id = None
		model_manager.state = ModelState.IDLE
		model_manager.cancel_token = None
		model_manager.loading_task = None

	def test_get_state_returns_current_state(self):
		"""Test that get_state returns the current state."""
		model_manager.state = ModelState.LOADING
		assert model_manager.get_state() == ModelState.LOADING

		model_manager.state = ModelState.LOADED
		assert model_manager.get_state() == ModelState.LOADED

	def test_set_state_updates_state(self):
		"""Test that set_state updates the state."""
		assert model_manager.state == ModelState.IDLE
		model_manager.set_state(ModelState.LOADING, '(test)')
		assert model_manager.state == ModelState.LOADING

	def test_set_state_logs_transition(self, caplog):
		"""Test that set_state logs state transitions."""
		import logging

		caplog.set_level(logging.INFO)
		model_manager.set_state(ModelState.LOADING, '(test)')
		assert 'Model state transition: idle -> loading (test)' in caplog.text


class TestReleaseResources:
	"""Test release_resources() method."""

	def setup_method(self):
		"""Reset model_manager state before each test."""
		model_manager.pipe = None
		model_manager.id = None
		model_manager.state = ModelState.IDLE

	def test_release_resources_clears_pipe_and_id(self):
		"""Test that release_resources clears pipe and id."""
		mock_pipe = MagicMock()
		model_manager.pipe = mock_pipe
		model_manager.id = 'test/model'

		model_manager.release_resources()

		assert model_manager.pipe is None
		assert model_manager.id is None

	@patch('app.cores.model_manager.model_manager.device_service')
	@patch('app.cores.model_manager.model_manager.torch')
	@patch('app.cores.model_manager.model_manager.gc')
	def test_release_resources_with_cuda(self, mock_gc, mock_torch, mock_device):
		"""Test release_resources with CUDA device."""
		# Setup
		mock_device.is_available = True
		mock_device.is_cuda = True
		mock_torch.cuda.memory_allocated.side_effect = [
			10 * (1024**3),  # Before: 10GB
			2 * (1024**3),  # After: 2GB
		]
		mock_torch.cuda.memory_reserved.side_effect = [
			12 * (1024**3),  # Before: 12GB
			3 * (1024**3),  # After: 3GB
		]

		mock_pipe = MagicMock()
		model_manager.pipe = mock_pipe
		model_manager.id = 'test/model'

		# Execute
		model_manager.release_resources()

		# Verify CUDA operations
		mock_torch.cuda.synchronize.assert_called_once()
		mock_torch.cuda.empty_cache.assert_called_once()
		assert mock_gc.collect.call_count >= 2  # Multiple GC passes

	@patch('app.cores.model_manager.model_manager.device_service')
	@patch('app.cores.model_manager.model_manager.torch')
	@patch('app.cores.model_manager.model_manager.gc')
	def test_release_resources_with_mps(self, mock_gc, mock_torch, mock_device):
		"""Test release_resources with MPS device."""
		# Setup
		mock_device.is_available = True
		mock_device.is_cuda = False
		mock_device.is_mps = True

		mock_pipe = MagicMock()
		model_manager.pipe = mock_pipe

		# Execute
		model_manager.release_resources()

		# Verify MPS operations
		mock_torch.mps.synchronize.assert_called_once()
		mock_torch.mps.empty_cache.assert_called_once()
		assert mock_gc.collect.call_count >= 2

	@patch('app.cores.model_manager.model_manager.device_service')
	@patch('app.cores.model_manager.model_manager.gc')
	def test_release_resources_with_no_gpu(self, mock_gc, mock_device):
		"""Test release_resources when no GPU is available."""
		# Setup
		mock_device.is_available = False

		mock_pipe = MagicMock()
		model_manager.pipe = mock_pipe

		# Execute
		model_manager.release_resources()

		# Verify only GC runs, no GPU operations
		assert mock_gc.collect.call_count >= 2

	def test_release_resources_handles_none_pipe(self):
		"""Test release_resources when pipe is already None."""
		model_manager.pipe = None
		model_manager.id = None

		# Should not raise
		model_manager.release_resources()


class TestLoadModel:
	"""Test load_model() synchronous method."""

	def setup_method(self):
		"""Reset model_manager state before each test."""
		model_manager.pipe = None
		model_manager.id = None
		model_manager.state = ModelState.IDLE
		model_manager.cancel_token = None
		model_manager.loading_task = None

	@patch('app.cores.model_manager.model_manager.socket_service')
	@patch('app.cores.model_manager.model_manager.model_loader')
	def test_load_model_when_already_loaded(self, mock_loader, mock_socket):
		"""Test load_model returns cached config when model already loaded."""
		# Setup - model already loaded
		mock_pipe = MagicMock()
		mock_pipe.config = {'model_type': 'test'}
		mock_pipe.unet.config = MagicMock()

		model_manager.pipe = mock_pipe
		model_manager.id = 'test/model'
		model_manager.state = ModelState.LOADED

		# Execute
		result = model_manager.load_model('test/model')

		# Verify
		assert result == {'model_type': 'test'}
		mock_loader.assert_not_called()  # Should not load again
		mock_socket.model_load_completed.assert_called_once()

	@patch('app.cores.model_manager.model_manager.model_loader')
	def test_load_model_success(self, mock_loader):
		"""Test successful model loading."""
		# Setup
		mock_pipe = MagicMock()
		mock_pipe.config = {'model_type': 'test', 'model_id': 'new/model'}
		mock_loader.return_value = mock_pipe

		# Execute
		result = model_manager.load_model('new/model')

		# Verify
		assert result == {'model_type': 'test', 'model_id': 'new/model'}
		assert model_manager.pipe == mock_pipe
		assert model_manager.id == 'new/model'
		mock_loader.assert_called_once()

	@patch('app.cores.model_manager.model_manager.model_loader')
	def test_load_model_unloads_existing_before_loading_new(self, mock_loader):
		"""Test that load_model unloads existing model before loading new one."""
		# Setup - existing model loaded
		old_pipe = MagicMock()
		model_manager.pipe = old_pipe
		model_manager.id = 'old/model'
		model_manager.state = ModelState.IDLE

		new_pipe = MagicMock()
		new_pipe.config = {'model_id': 'new/model'}
		mock_loader.return_value = new_pipe

		with patch.object(model_manager, 'unload_internal') as mock_unload:
			# Execute
			model_manager.load_model('new/model')

			# Verify old model was unloaded
			mock_unload.assert_called_once()
			assert model_manager.id == 'new/model'

	@patch('app.cores.model_manager.model_manager.model_loader')
	def test_load_model_handles_cancellation(self, mock_loader):
		"""Test load_model handles CancellationException."""
		from app.cores.model_loader.cancellation import CancellationException

		mock_loader.side_effect = CancellationException('Cancelled')

		with patch.object(model_manager, 'unload_internal') as mock_unload:
			with pytest.raises(CancellationException):
				model_manager.load_model('test/model')

			# Verify cleanup happened
			mock_unload.assert_called_once()

	@patch('app.cores.model_manager.model_manager.model_loader')
	def test_load_model_handles_error(self, mock_loader):
		"""Test load_model handles generic errors."""
		mock_loader.side_effect = ValueError('Load failed')

		with patch.object(model_manager, 'unload_internal') as mock_unload:
			with pytest.raises(ValueError, match='Load failed'):
				model_manager.load_model('test/model')

			# Verify cleanup happened
			mock_unload.assert_called_once()


class TestUnloadModel:
	"""Test unload_model() synchronous method."""

	def setup_method(self):
		"""Reset model_manager state before each test."""
		model_manager.pipe = None
		model_manager.id = None
		model_manager.state = ModelState.IDLE

	def test_unload_model_when_model_loaded(self):
		"""Test unload_model successfully unloads model."""
		# Setup
		mock_pipe = MagicMock()
		model_manager.pipe = mock_pipe
		model_manager.id = 'test/model'
		model_manager.state = ModelState.LOADED

		with patch.object(model_manager, 'unload_internal') as mock_unload:
			# Execute
			model_manager.unload_model()

			# Verify
			mock_unload.assert_called_once()
			assert model_manager.state == ModelState.IDLE

	def test_unload_model_when_no_model_loaded(self, caplog):
		"""Test unload_model when no model is loaded."""
		import logging

		# Setup
		model_manager.pipe = None
		model_manager.state = ModelState.IDLE

		caplog.set_level(logging.INFO)

		# Execute
		model_manager.unload_model()

		# Verify - should log and not error
		assert 'No model loaded, nothing to unload.' in caplog.text

	def test_unload_model_handles_error(self):
		"""Test unload_model handles errors during unload."""
		# Setup
		mock_pipe = MagicMock()
		model_manager.pipe = mock_pipe
		model_manager.id = 'test/model'
		model_manager.state = ModelState.LOADED

		with patch.object(model_manager, 'unload_internal', side_effect=RuntimeError('Unload failed')):
			# Execute - should not raise
			model_manager.unload_model()

			# Verify state transitions to ERROR
			assert model_manager.state == ModelState.ERROR


class TestSetSampler:
	"""Test set_sampler() method."""

	def setup_method(self):
		"""Reset model_manager state before each test."""
		model_manager.pipe = None
		model_manager.id = None
		model_manager.state = ModelState.IDLE

	def test_set_sampler_raises_when_no_model_loaded(self):
		"""Test set_sampler raises ValueError when no model loaded."""
		model_manager.pipe = None
		model_manager.state = ModelState.IDLE

		with pytest.raises(ValueError, match='No model loaded or model not in LOADED state'):
			model_manager.set_sampler(SamplerType.EULER)

	def test_set_sampler_raises_when_not_in_loaded_state(self):
		"""Test set_sampler raises ValueError when state is not LOADED."""
		mock_pipe = MagicMock()
		model_manager.pipe = mock_pipe
		model_manager.state = ModelState.LOADING  # Not LOADED

		with pytest.raises(ValueError, match='No model loaded or model not in LOADED state'):
			model_manager.set_sampler(SamplerType.EULER)

	def test_set_sampler_success(self):
		"""Test successful sampler change."""
		# Setup
		mock_scheduler_config = {'key': 'value'}
		mock_scheduler = MagicMock()
		mock_scheduler.config = mock_scheduler_config

		mock_pipe = MagicMock()
		mock_pipe.scheduler = mock_scheduler

		model_manager.pipe = mock_pipe
		model_manager.state = ModelState.LOADED

		# Mock the SCHEDULER_MAPPING
		mock_new_scheduler_class = MagicMock()
		mock_new_scheduler_instance = MagicMock()
		mock_new_scheduler_class.from_config.return_value = mock_new_scheduler_instance

		with patch(
			'app.cores.model_manager.model_manager.SCHEDULER_MAPPING', {SamplerType.EULER: mock_new_scheduler_class}
		):
			# Execute
			model_manager.set_sampler(SamplerType.EULER)

			# Verify
			mock_new_scheduler_class.from_config.assert_called_once_with(mock_scheduler_config)
			assert model_manager.pipe.scheduler == mock_new_scheduler_instance

	def test_set_sampler_with_karras_sigma(self):
		"""Test set_sampler with Karras sigma samplers."""
		# Setup
		mock_scheduler_config = {'key': 'value'}
		mock_scheduler = MagicMock()
		mock_scheduler.config = mock_scheduler_config

		mock_pipe = MagicMock()
		mock_pipe.scheduler = mock_scheduler

		model_manager.pipe = mock_pipe
		model_manager.state = ModelState.LOADED

		mock_new_scheduler_class = MagicMock()
		mock_new_scheduler_instance = MagicMock()
		mock_new_scheduler_class.from_config.return_value = mock_new_scheduler_instance

		with patch(
			'app.cores.model_manager.model_manager.SCHEDULER_MAPPING',
			{SamplerType.DPM_SOLVER_MULTISTEP_KARRAS: mock_new_scheduler_class},
		):
			# Execute
			model_manager.set_sampler(SamplerType.DPM_SOLVER_MULTISTEP_KARRAS)

			# Verify use_karras_sigmas was passed
			mock_new_scheduler_class.from_config.assert_called_once_with(mock_scheduler_config, use_karras_sigmas=True)


class TestGetSampleSize:
	"""Test get_sample_size() method."""

	def setup_method(self):
		"""Reset model_manager state before each test."""
		model_manager.pipe = None
		model_manager.id = None
		model_manager.state = ModelState.IDLE

	def test_get_sample_size_raises_when_no_model_loaded(self):
		"""Test get_sample_size raises ValueError when no model loaded."""
		model_manager.pipe = None
		model_manager.state = ModelState.IDLE

		with pytest.raises(ValueError, match='No model loaded or model not in LOADED state'):
			model_manager.get_sample_size()

	def test_get_sample_size_raises_when_not_in_loaded_state(self):
		"""Test get_sample_size raises ValueError when state is not LOADED."""
		mock_pipe = MagicMock()
		model_manager.pipe = mock_pipe
		model_manager.state = ModelState.LOADING

		with pytest.raises(ValueError, match='No model loaded or model not in LOADED state'):
			model_manager.get_sample_size()

	def test_get_sample_size_returns_model_sample_size(self):
		"""Test get_sample_size returns sample_size from unet config."""
		# Setup
		mock_unet_config = MagicMock()
		mock_unet_config.sample_size = 64

		mock_pipe = MagicMock()
		mock_pipe.unet.config = mock_unet_config

		model_manager.pipe = mock_pipe
		model_manager.state = ModelState.LOADED

		# Execute
		result = model_manager.get_sample_size()

		# Verify
		assert result == 64

	def test_get_sample_size_returns_default_when_no_sample_size(self):
		"""Test get_sample_size returns DEFAULT_SAMPLE_SIZE when not in config."""
		from app.cores.constants.samplers import DEFAULT_SAMPLE_SIZE

		# Setup - config without sample_size attribute
		# Use spec_set to control which attributes exist
		mock_unet_config = MagicMock(spec_set=['other_attr'])

		mock_pipe = MagicMock()
		mock_pipe.unet.config = mock_unet_config

		model_manager.pipe = mock_pipe
		model_manager.state = ModelState.LOADED

		# Execute
		result = model_manager.get_sample_size()

		# Verify
		assert result == DEFAULT_SAMPLE_SIZE


class TestUnloadInternal:
	"""Test unload_internal() method."""

	def setup_method(self):
		"""Reset model_manager state before each test."""
		model_manager.pipe = None
		model_manager.id = None
		model_manager.state = ModelState.IDLE

	def test_unload_internal_calls_release_resources(self):
		"""Test that unload_internal calls release_resources when pipe is set."""
		mock_pipe = MagicMock()
		model_manager.pipe = mock_pipe
		model_manager.id = 'test/model'

		with patch.object(model_manager, 'release_resources') as mock_release:
			# Execute
			model_manager.unload_internal()

			# Verify
			mock_release.assert_called_once()

	def test_unload_internal_does_nothing_when_no_pipe(self, caplog):
		"""Test that unload_internal does nothing when pipe is None."""
		import logging

		model_manager.pipe = None
		caplog.set_level(logging.INFO)

		with patch.object(model_manager, 'release_resources') as mock_release:
			# Execute
			model_manager.unload_internal()

			# Verify - should not call release_resources
			mock_release.assert_not_called()


class TestAsyncMethods:
	"""Test async methods of ModelManager."""

	def setup_method(self):
		"""Reset model_manager state before each test."""
		model_manager.pipe = None
		model_manager.id = None
		model_manager.state = ModelState.IDLE
		model_manager.cancel_token = None
		model_manager.loading_task = None

	@pytest.mark.asyncio
	async def test_load_model_async_raises_error_on_exception(self):
		"""Test load_model_async error handling (lines 302-309)."""
		with patch.object(model_manager, 'load_model', side_effect=ValueError('Load failed')):
			with pytest.raises(ValueError, match='Load failed'):
				await model_manager.load_model_async('test/model')

			# Verify state transitioned to ERROR
			assert model_manager.state == ModelState.ERROR

	@pytest.mark.asyncio
	async def test_unload_model_async_when_idle(self, caplog):
		"""Test unload_model_async when state is IDLE (lines 349-351)."""
		import logging

		model_manager.state = ModelState.IDLE
		caplog.set_level(logging.INFO)

		# Execute
		await model_manager.unload_model_async()

		# Verify - should log and return early
		assert 'No model loaded, nothing to unload' in caplog.text

	@pytest.mark.asyncio
	async def test_unload_model_async_when_loaded(self):
		"""Test unload_model_async when state is LOADED (lines 353-365)."""
		# Setup
		mock_pipe = MagicMock()
		model_manager.pipe = mock_pipe
		model_manager.id = 'test/model'
		model_manager.state = ModelState.LOADED

		with patch.object(model_manager, 'unload_internal') as mock_unload:
			# Execute
			await model_manager.unload_model_async()

			# Verify
			mock_unload.assert_called_once()
			assert model_manager.state == ModelState.IDLE

	@pytest.mark.asyncio
	async def test_unload_model_async_handles_error(self):
		"""Test unload_model_async handles exceptions (lines 362-365)."""
		# Setup
		mock_pipe = MagicMock()
		model_manager.pipe = mock_pipe
		model_manager.state = ModelState.LOADED

		with patch.object(model_manager, 'unload_internal', side_effect=RuntimeError('Unload failed')):
			with pytest.raises(RuntimeError, match='Unload failed'):
				await model_manager.unload_model_async()

			# Verify state transitioned to ERROR
			assert model_manager.state == ModelState.ERROR

	@pytest.mark.asyncio
	async def test_unload_model_async_resets_from_error_state(self):
		"""Test unload_model_async resets from ERROR state (lines 367-371)."""
		# Setup
		model_manager.state = ModelState.ERROR
		model_manager.pipe = None

		with patch.object(model_manager, 'unload_internal') as mock_unload:
			# Execute
			await model_manager.unload_model_async()

			# Verify
			mock_unload.assert_called_once()
			assert model_manager.state == ModelState.IDLE

	@pytest.mark.asyncio
	async def test_unload_model_async_resets_from_cancelling_state(self):
		"""Test unload_model_async resets from CANCELLING state (lines 367-371)."""
		# Setup
		model_manager.state = ModelState.CANCELLING
		model_manager.pipe = None

		with patch.object(model_manager, 'unload_internal') as mock_unload:
			# Execute
			await model_manager.unload_model_async()

			# Verify
			mock_unload.assert_called_once()
			assert model_manager.state == ModelState.IDLE


class TestSetSamplerEdgeCases:
	"""Test edge cases for set_sampler()."""

	def setup_method(self):
		"""Reset model_manager state before each test."""
		model_manager.pipe = None
		model_manager.id = None
		model_manager.state = ModelState.IDLE

	def test_set_sampler_raises_for_unsupported_sampler(self):
		"""Test set_sampler raises ValueError for unsupported sampler (line 409)."""
		# Setup
		mock_pipe = MagicMock()
		mock_pipe.scheduler.config = {}
		model_manager.pipe = mock_pipe
		model_manager.state = ModelState.LOADED

		# Create a fake sampler type that's not in SCHEDULER_MAPPING
		from app.cores.samplers import SamplerType

		# Mock SCHEDULER_MAPPING to be empty
		with patch('app.cores.model_manager.model_manager.SCHEDULER_MAPPING', {}):
			with pytest.raises(ValueError, match='Unsupported sampler type'):
				model_manager.set_sampler(SamplerType.EULER)
