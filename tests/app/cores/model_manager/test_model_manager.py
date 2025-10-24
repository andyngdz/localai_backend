"""Integration tests for ModelManager facade.

This test suite focuses on testing the ModelManager facade's delegation
to specialized managers (StateManager, ResourceManager, PipelineManager, LoaderService).

Component-specific tests are in separate files:
- test_state_manager.py - StateManager tests
- test_resource_manager.py - ResourceManager tests
- test_pipeline_manager.py - PipelineManager tests
- test_loader_service.py - LoaderService tests
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.cores.model_manager import ModelManager, ModelState, StateTransitionReason
from app.cores.samplers import SamplerType


class TestModelManagerInitialization:
	"""Test ModelManager initialization and architecture."""

	def test_model_manager_initializes_all_managers(self):
		"""Test ModelManager creates all internal manager instances."""
		model_manager = ModelManager()

		# Verify all managers are initialized
		assert model_manager.state_manager is not None
		assert model_manager.resource_manager is not None
		assert model_manager.pipeline_manager is not None
		assert model_manager.loader_service is not None

	def test_model_manager_managers_are_public(self):
		"""Test that internal managers are public (no underscores)."""
		model_manager = ModelManager()

		# Verify managers are accessible (not private with _)
		assert hasattr(model_manager, 'state_manager')
		assert hasattr(model_manager, 'resource_manager')
		assert hasattr(model_manager, 'pipeline_manager')
		assert hasattr(model_manager, 'loader_service')


class TestModelManagerDelegation:
	"""Test ModelManager delegates to specialized managers."""

	def setup_method(self):
		"""Create fresh ModelManager for each test."""
		self.model_manager = ModelManager()

	@pytest.mark.asyncio
	async def test_load_model_async_delegates_to_loader_service(self):
		"""Test load_model_async delegates to LoaderService."""
		mock_config = {'model_id': 'test/model'}

		with patch.object(self.model_manager.loader_service, 'load_model_async', return_value=mock_config) as mock_load:
			# Execute
			result = await self.model_manager.load_model_async('test/model')

			# Verify delegation
			mock_load.assert_called_once_with('test/model')
			assert result == mock_config

	@pytest.mark.asyncio
	async def test_unload_model_async_delegates_to_loader_service(self):
		"""Test unload_model_async delegates to LoaderService."""
		with patch.object(self.model_manager.loader_service, 'unload_model_async', new_callable=AsyncMock) as mock_unload:
			# Execute
			await self.model_manager.unload_model_async()

			# Verify delegation
			mock_unload.assert_called_once()

	def test_current_state_property_delegates_to_state_manager(self):
		"""Test current_state property delegates to StateManager."""
		self.model_manager.state_manager._state = ModelState.LOADED

		# Execute
		result = self.model_manager.current_state

		# Verify
		assert result == ModelState.LOADED

	def test_set_state_delegates_to_state_manager(self):
		"""Test set_state delegates to StateManager."""
		with patch.object(self.model_manager.state_manager, 'set_state') as mock_set:
			# Execute
			self.model_manager.set_state(ModelState.LOADING, StateTransitionReason.LOAD_REQUESTED)

			# Verify delegation
			mock_set.assert_called_once_with(ModelState.LOADING, StateTransitionReason.LOAD_REQUESTED)

	def test_set_sampler_delegates_to_pipeline_manager(self):
		"""Test set_sampler delegates to PipelineManager."""
		with patch.object(self.model_manager.pipeline_manager, 'set_sampler') as mock_set:
			# Execute
			self.model_manager.set_sampler(SamplerType.EULER)

			# Verify delegation
			mock_set.assert_called_once_with(SamplerType.EULER)

	def test_sample_size_property_delegates_to_pipeline_manager(self):
		"""Test sample_size property delegates to PipelineManager."""
		with patch.object(self.model_manager.pipeline_manager, 'get_sample_size', return_value=64) as mock_get:
			# Execute
			result = self.model_manager.sample_size

			# Verify delegation
			mock_get.assert_called_once()
			assert result == 64


class TestModelManagerBackwardCompatibility:
	"""Test backward compatibility properties (pipe, id, state, lock)."""

	def setup_method(self):
		"""Create fresh ModelManager for each test."""
		self.model_manager = ModelManager()

	def test_pipe_property_getter_delegates_to_pipeline_manager(self):
		"""Test pipe property getter delegates to PipelineManager.get_pipeline()."""
		mock_pipe = MagicMock()

		with patch.object(self.model_manager.pipeline_manager, 'get_pipeline', return_value=mock_pipe) as mock_get:
			# Execute
			result = self.model_manager.pipe

			# Verify delegation
			mock_get.assert_called_once()
			assert result == mock_pipe

	def test_pipe_property_setter_delegates_to_pipeline_manager(self):
		"""Test pipe property setter delegates to PipelineManager.set_pipeline()."""
		mock_pipe = MagicMock()

		# Setup - model_id must exist
		self.model_manager.pipeline_manager.model_id = 'test/model'

		with patch.object(self.model_manager.pipeline_manager, 'set_pipeline') as mock_set:
			# Execute
			self.model_manager.pipe = mock_pipe

			# Verify delegation
			mock_set.assert_called_once_with(mock_pipe, 'test/model')

	def test_pipe_property_setter_raises_when_no_model_id(self):
		"""Test pipe property setter raises ValueError when no model_id exists."""
		mock_pipe = MagicMock()

		# Setup - no model_id
		self.model_manager.pipeline_manager.model_id = None

		# Execute & Verify
		with pytest.raises(ValueError, match='Cannot set pipeline without loading a model first'):
			self.model_manager.pipe = mock_pipe

	def test_id_property_getter_delegates_to_pipeline_manager(self):
		"""Test id property getter delegates to PipelineManager.get_model_id()."""
		with patch.object(self.model_manager.pipeline_manager, 'get_model_id', return_value='test/model') as mock_get:
			# Execute
			result = self.model_manager.id

			# Verify delegation
			mock_get.assert_called_once()
			assert result == 'test/model'

	def test_id_property_setter_sets_pipeline_manager_model_id(self):
		"""Test id property setter directly sets pipeline_manager.model_id."""
		# Execute
		self.model_manager.id = 'new/model'

		# Verify
		assert self.model_manager.pipeline_manager.model_id == 'new/model'

	def test_current_state_property_delegates_to_state_manager(self):
		"""Test current_state property delegates to StateManager.current_state."""
		# Setup
		self.model_manager.state_manager._state = ModelState.LOADED

		# Execute
		result = self.model_manager.current_state

		# Verify
		assert result == ModelState.LOADED

	def test_sample_size_property_delegates_to_pipeline_manager(self):
		"""Test sample_size property delegates to PipelineManager.get_sample_size()."""
		with patch.object(self.model_manager.pipeline_manager, 'get_sample_size', return_value=64) as mock_get:
			# Execute
			result = self.model_manager.sample_size

			# Verify delegation
			mock_get.assert_called_once()
			assert result == 64

	def test_lock_property_exposes_loader_service_lock(self):
		"""Test lock property exposes LoaderService lock for external synchronization."""
		# Execute
		result = self.model_manager.lock

		# Verify
		assert result is self.model_manager.loader_service.lock


class TestModelManagerIntegration:
	"""Integration tests for complete workflows through the facade."""

	def setup_method(self):
		"""Create fresh ModelManager for each test."""
		self.model_manager = ModelManager()

	@pytest.mark.asyncio
	async def test_load_model_async_integration(self):
		"""Test complete load flow through ModelManager facade."""
		mock_pipe = MagicMock()
		mock_pipe.config = {'model_id': 'test/model'}

		with patch('app.cores.model_manager.loader_service.model_loader', return_value=mock_pipe):
			with patch('app.cores.model_manager.loader_service.socket_service'):
				# Execute
				result = await self.model_manager.load_model_async('test/model')

				# Verify state transitions and pipeline storage
				assert self.model_manager.current_state == ModelState.LOADED
				assert self.model_manager.pipeline_manager.get_model_id() == 'test/model'
				assert result == {'model_id': 'test/model'}

	@pytest.mark.asyncio
	async def test_unload_model_async_integration(self):
		"""Test complete unload flow through ModelManager facade."""
		# Setup - load a model first
		mock_pipe = MagicMock()
		self.model_manager.pipeline_manager.set_pipeline(mock_pipe, 'test/model')
		self.model_manager.state_manager.set_state(ModelState.LOADED, StateTransitionReason.LOAD_COMPLETED)

		with patch.object(self.model_manager.resource_manager, 'cleanup_pipeline'):
			# Execute
			await self.model_manager.unload_model_async()

			# Verify state and pipeline cleared
			assert self.model_manager.current_state == ModelState.IDLE
			assert self.model_manager.pipeline_manager.get_pipeline() is None
			assert self.model_manager.pipeline_manager.get_model_id() is None

	def test_set_sampler_integration(self):
		"""Test set_sampler works through ModelManager facade."""
		# Setup - model must be loaded
		mock_scheduler = MagicMock()
		mock_scheduler.config = {'key': 'value'}

		mock_pipe = MagicMock()
		mock_pipe.scheduler = mock_scheduler

		self.model_manager.pipeline_manager.set_pipeline(mock_pipe, 'test/model')

		# Mock SCHEDULER_MAPPING
		mock_new_scheduler_class = MagicMock()
		mock_new_scheduler_instance = MagicMock()
		mock_new_scheduler_class.from_config.return_value = mock_new_scheduler_instance

		with patch(
			'app.cores.model_manager.pipeline_manager.SCHEDULER_MAPPING', {SamplerType.EULER: mock_new_scheduler_class}
		):
			# Execute
			self.model_manager.set_sampler(SamplerType.EULER)

			# Verify scheduler changed
			assert self.model_manager.pipeline_manager.pipe.scheduler == mock_new_scheduler_instance

	def test_sample_size_integration(self):
		"""Test sample_size property works through ModelManager facade."""
		# Setup
		mock_unet_config = MagicMock()
		mock_unet_config.sample_size = 64

		mock_pipe = MagicMock()
		mock_pipe.unet.config = mock_unet_config

		self.model_manager.pipeline_manager.set_pipeline(mock_pipe, 'test/model')

		# Execute
		result = self.model_manager.sample_size

		# Verify
		assert result == 64


class TestModelManagerSingletonInstance:
	"""Test the singleton model_manager instance."""

	def test_singleton_instance_exists(self):
		"""Test that a singleton model_manager instance is exported."""
		from app.cores.model_manager import model_manager

		assert model_manager is not None
		assert isinstance(model_manager, ModelManager)

	def test_singleton_has_all_managers(self):
		"""Test singleton instance has all internal managers."""
		from app.cores.model_manager import model_manager

		assert hasattr(model_manager, 'state_manager')
		assert hasattr(model_manager, 'resource_manager')
		assert hasattr(model_manager, 'pipeline_manager')
		assert hasattr(model_manager, 'loader_service')
