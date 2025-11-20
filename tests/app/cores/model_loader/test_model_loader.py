"""Tests for the model loader module."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from app.cores.model_loader.cancellation import CancellationException, CancellationToken
from app.cores.model_loader.model_loader import (
	apply_device_optimizations,
	cleanup_partial_load,
	emit_progress,
	find_checkpoint_in_cache,
	find_single_file_checkpoint,
	map_step_to_phase,
	model_loader,
	move_to_device,
)
from app.cores.model_loader.schemas import ModelLoadPhase


class TestMapStepToPhase:
	"""Test the map_step_to_phase function."""

	def test_initialization_phase(self) -> None:
		"""Test that steps 1-2 map to INITIALIZATION."""
		assert map_step_to_phase(1) == ModelLoadPhase.INITIALIZATION
		assert map_step_to_phase(2) == ModelLoadPhase.INITIALIZATION

	def test_loading_model_phase(self) -> None:
		"""Test that steps 3-5 map to LOADING_MODEL."""
		assert map_step_to_phase(3) == ModelLoadPhase.LOADING_MODEL
		assert map_step_to_phase(4) == ModelLoadPhase.LOADING_MODEL
		assert map_step_to_phase(5) == ModelLoadPhase.LOADING_MODEL

	def test_device_setup_phase(self) -> None:
		"""Test that steps 6-7 map to DEVICE_SETUP."""
		assert map_step_to_phase(6) == ModelLoadPhase.DEVICE_SETUP
		assert map_step_to_phase(7) == ModelLoadPhase.DEVICE_SETUP

	def test_optimization_phase(self) -> None:
		"""Test that steps 8+ map to OPTIMIZATION."""
		assert map_step_to_phase(8) == ModelLoadPhase.OPTIMIZATION
		assert map_step_to_phase(9) == ModelLoadPhase.OPTIMIZATION


class TestFindSingleFileCheckpoint:
	"""Test the find_single_file_checkpoint function."""

	def test_finds_safetensors_file(self, tmp_path: Path) -> None:
		"""Test that .safetensors file is found in directory."""
		model_dir = tmp_path / 'model'
		model_dir.mkdir()
		checkpoint_file = model_dir / 'model.safetensors'
		checkpoint_file.touch()

		result = find_single_file_checkpoint(str(model_dir))

		assert result == str(checkpoint_file)

	def test_returns_none_when_directory_not_exists(self) -> None:
		"""Test that None is returned when directory doesn't exist."""
		result = find_single_file_checkpoint('/nonexistent/path')

		assert result is None

	def test_returns_none_when_no_safetensors_files(self, tmp_path: Path) -> None:
		"""Test that None is returned when no .safetensors files exist."""
		model_dir = tmp_path / 'model'
		model_dir.mkdir()
		# Create non-safetensors files
		(model_dir / 'config.json').touch()
		(model_dir / 'model.bin').touch()

		result = find_single_file_checkpoint(str(model_dir))

		assert result is None

	def test_returns_first_safetensors_when_multiple_exist(self, tmp_path: Path) -> None:
		"""Test that first .safetensors file is returned when multiple exist."""
		model_dir = tmp_path / 'model'
		model_dir.mkdir()
		checkpoint1 = model_dir / 'model1.safetensors'
		checkpoint2 = model_dir / 'model2.safetensors'
		checkpoint1.touch()
		checkpoint2.touch()

		result = find_single_file_checkpoint(str(model_dir))

		# Should return one of the checkpoint files
		assert result is not None
		assert result.endswith('.safetensors')


class TestFindCheckpointInCache:
	"""Test the find_checkpoint_in_cache function."""

	def test_finds_checkpoint_in_snapshot(self, tmp_path: Path) -> None:
		"""Test finding checkpoint in HuggingFace cache structure."""
		cache_dir = tmp_path / 'cache'
		snapshots_dir = cache_dir / 'snapshots'
		snapshot_dir = snapshots_dir / 'abc123'
		snapshot_dir.mkdir(parents=True)
		checkpoint_file = snapshot_dir / 'model.safetensors'
		checkpoint_file.touch()

		result = find_checkpoint_in_cache(str(cache_dir))

		assert result == str(checkpoint_file)

	def test_returns_none_when_cache_not_exists(self) -> None:
		"""Test that None is returned when cache directory doesn't exist."""
		result = find_checkpoint_in_cache('/nonexistent/cache')

		assert result is None

	def test_returns_none_when_snapshots_dir_missing(self, tmp_path: Path) -> None:
		"""Test that None is returned when snapshots directory is missing."""
		cache_dir = tmp_path / 'cache'
		cache_dir.mkdir()

		result = find_checkpoint_in_cache(str(cache_dir))

		assert result is None

	def test_returns_none_when_no_snapshots_exist(self, tmp_path: Path) -> None:
		"""Test that None is returned when snapshots directory is empty."""
		cache_dir = tmp_path / 'cache'
		snapshots_dir = cache_dir / 'snapshots'
		snapshots_dir.mkdir(parents=True)

		result = find_checkpoint_in_cache(str(cache_dir))

		assert result is None

	def test_returns_none_when_snapshot_has_no_checkpoint(self, tmp_path: Path) -> None:
		"""Test that None is returned when snapshot has no .safetensors file."""
		cache_dir = tmp_path / 'cache'
		snapshots_dir = cache_dir / 'snapshots'
		snapshot_dir = snapshots_dir / 'abc123'
		snapshot_dir.mkdir(parents=True)
		# Create non-checkpoint files
		(snapshot_dir / 'config.json').touch()

		result = find_checkpoint_in_cache(str(cache_dir))

		assert result is None


class TestEmitProgress:
	"""Test the emit_progress function."""

	@patch('app.cores.model_loader.model_loader.socket_service')
	def test_emits_progress_successfully(self, mock_socket_service: Mock) -> None:
		"""Test that progress is emitted successfully."""
		emit_progress('test-model', 5, 'Loading model...')

		mock_socket_service.model_load_progress.assert_called_once()

	@patch('app.cores.model_loader.model_loader.socket_service')
	def test_handles_exception_gracefully(self, mock_socket_service: Mock) -> None:
		"""Test that emit_progress handles exceptions gracefully."""
		mock_socket_service.model_load_progress.side_effect = Exception('Socket error')

		# Should not raise - exception should be caught and logged
		emit_progress('test-model', 5, 'Loading model...')

		mock_socket_service.model_load_progress.assert_called_once()


class TestApplyDeviceOptimizations:
	"""Test the apply_device_optimizations function."""

	@patch('app.cores.model_loader.model_loader.get_optimizer')
	def test_applies_optimizer_to_pipeline(self, mock_get_optimizer: Mock) -> None:
		"""Test that platform optimizer is applied to pipeline."""
		mock_optimizer = Mock()
		mock_optimizer.get_platform_name.return_value = 'TestPlatform'
		mock_get_optimizer.return_value = mock_optimizer
		mock_pipe = Mock()

		apply_device_optimizations(mock_pipe)

		mock_optimizer.apply.assert_called_once_with(mock_pipe)
		mock_optimizer.get_platform_name.assert_called_once()


class TestMoveToDevice:
	"""Test the move_to_device function."""

	def test_uses_to_empty_when_available(self) -> None:
		"""Test that to_empty() is used when available."""
		mock_pipe = Mock()
		mock_pipe.to_empty.return_value = mock_pipe

		result = move_to_device(mock_pipe, 'cuda', 'Test pipeline')

		mock_pipe.to_empty.assert_called_once_with('cuda')
		mock_pipe.to.assert_not_called()
		assert result is mock_pipe

	def test_falls_back_to_to_when_to_empty_not_available(self) -> None:
		"""Test fallback to to() when to_empty() raises AttributeError."""
		mock_pipe = Mock()
		mock_pipe.to_empty.side_effect = AttributeError('to_empty not available')
		mock_pipe.to.return_value = mock_pipe

		result = move_to_device(mock_pipe, 'cuda', 'Test pipeline')

		mock_pipe.to.assert_called_once_with('cuda')
		assert result is mock_pipe

	def test_falls_back_to_to_when_to_empty_raises_type_error(self) -> None:
		"""Test fallback to to() when to_empty() raises TypeError."""
		mock_pipe = Mock()
		mock_pipe.to_empty.side_effect = TypeError('Invalid type')
		mock_pipe.to.return_value = mock_pipe

		result = move_to_device(mock_pipe, 'cpu', 'Test pipeline')

		mock_pipe.to.assert_called_once_with('cpu')
		assert result is mock_pipe


class TestCleanupPartialLoad:
	"""Test the cleanup_partial_load function."""

	def test_skips_cleanup_when_pipe_is_none(self) -> None:
		"""Test that cleanup is skipped when pipe is None."""
		# Should not raise any exception
		cleanup_partial_load(None)

	@patch('app.cores.model_loader.model_loader.cleanup_gpu_model')
	def test_cleans_up_pipeline_resources(self, mock_cleanup: Mock) -> None:
		"""Test that pipeline resources are cleaned up."""
		mock_metrics = Mock()
		mock_metrics.time_ms = 100.5
		mock_metrics.objects_collected = 42
		mock_metrics.error = None
		mock_cleanup.return_value = mock_metrics

		mock_pipe = Mock()

		cleanup_partial_load(mock_pipe)

		mock_cleanup.assert_called_once_with(mock_pipe, name='partial pipeline')

	@patch('app.cores.model_loader.model_loader.cleanup_gpu_model')
	def test_logs_cleanup_with_error(self, mock_cleanup: Mock) -> None:
		"""Test that cleanup errors are logged."""
		mock_metrics = Mock()
		mock_metrics.time_ms = 50.0
		mock_metrics.objects_collected = 10
		mock_metrics.error = 'Cleanup error'
		mock_cleanup.return_value = mock_metrics

		mock_pipe = Mock()

		cleanup_partial_load(mock_pipe)

		mock_cleanup.assert_called_once_with(mock_pipe, name='partial pipeline')


class TestModelLoader:
	"""Test the model_loader function."""

	@patch('app.cores.model_loader.model_loader.SessionLocal')
	@patch('app.cores.model_loader.model_loader.MaxMemoryConfig')
	@patch('app.cores.model_loader.model_loader.CLIPImageProcessor')
	@patch('app.cores.model_loader.model_loader.StableDiffusionSafetyChecker')
	@patch('app.cores.model_loader.model_loader.storage_service')
	@patch('app.cores.model_loader.model_loader.AutoPipelineForText2Image')
	@patch('app.cores.model_loader.model_loader.socket_service')
	@patch('app.cores.model_loader.model_loader.device_service')
	@patch('app.cores.model_loader.model_loader.apply_device_optimizations')
	@patch('app.cores.model_loader.model_loader.move_to_device')
	def test_loads_model_successfully(
		self,
		mock_move_to_device: Mock,
		mock_apply_optimizations: Mock,
		mock_device_service: Mock,
		mock_socket_service: Mock,
		mock_auto_pipeline: Mock,
		mock_storage_service: Mock,
		mock_safety_checker: Mock,
		mock_feature_extractor: Mock,
		mock_max_memory: Mock,
		mock_session_local: Mock,
	) -> None:
		"""Test successful model loading without cancellation."""
		# Setup mocks
		mock_db = Mock()
		mock_session_local.return_value = mock_db
		mock_max_memory.return_value.to_dict.return_value = {'cpu': '10GB'}
		mock_device_service.device = 'cuda'
		mock_device_service.torch_dtype = 'float16'
		mock_storage_service.get_model_dir.return_value = '/cache/model'

		mock_pipe = Mock()
		mock_pipe.reset_device_map = Mock()
		mock_auto_pipeline.from_pretrained.return_value = mock_pipe
		mock_move_to_device.return_value = mock_pipe

		# Execute
		result = model_loader('test-model')

		# Verify
		assert result is mock_pipe
		mock_socket_service.model_load_started.assert_called_once()
		mock_socket_service.model_load_completed.assert_called_once()
		mock_apply_optimizations.assert_called_once_with(mock_pipe)
		mock_db.close.assert_called_once()

	@patch('app.cores.model_loader.model_loader.SessionLocal')
	@patch('app.cores.model_loader.model_loader.MaxMemoryConfig')
	def test_raises_cancellation_exception_at_checkpoint_1(
		self,
		mock_max_memory: Mock,
		mock_session_local: Mock,
	) -> None:
		"""Test that cancellation at checkpoint 1 raises exception."""
		mock_db = Mock()
		mock_session_local.return_value = mock_db
		cancel_token = CancellationToken()
		cancel_token.cancel()

		with pytest.raises(CancellationException):
			model_loader('test-model', cancel_token=cancel_token)

		mock_db.close.assert_called_once()

	@patch('app.cores.model_loader.model_loader.SessionLocal')
	@patch('app.cores.model_loader.model_loader.MaxMemoryConfig')
	@patch('app.cores.model_loader.model_loader.CLIPImageProcessor')
	@patch('app.cores.model_loader.model_loader.StableDiffusionSafetyChecker')
	def test_raises_cancellation_exception_at_checkpoint_3(
		self,
		mock_safety_checker: Mock,
		mock_feature_extractor: Mock,
		mock_max_memory: Mock,
		mock_session_local: Mock,
	) -> None:
		"""Test that cancellation at checkpoint 3 raises exception."""
		mock_db = Mock()
		mock_session_local.return_value = mock_db
		mock_max_memory.return_value.to_dict.return_value = {}

		cancel_token = CancellationToken()

		# Let it pass checkpoint 1 and 2
		original_check = cancel_token.check_cancelled
		call_count = {'count': 0}

		def delayed_cancel():
			call_count['count'] += 1
			if call_count['count'] >= 3:
				cancel_token.cancel()
			original_check()

		setattr(cancel_token, 'check_cancelled', delayed_cancel)

		with pytest.raises(CancellationException):
			model_loader('test-model', cancel_token=cancel_token)

		mock_db.close.assert_called_once()

	@patch('app.cores.model_loader.model_loader.SessionLocal')
	@patch('app.cores.model_loader.model_loader.MaxMemoryConfig')
	@patch('app.cores.model_loader.model_loader.CLIPImageProcessor')
	@patch('app.cores.model_loader.model_loader.StableDiffusionSafetyChecker')
	@patch('app.cores.model_loader.model_loader.storage_service')
	@patch('app.cores.model_loader.model_loader.AutoPipelineForText2Image')
	@patch('app.cores.model_loader.model_loader.socket_service')
	@patch('app.cores.model_loader.model_loader.device_service')
	@patch('app.cores.model_loader.model_loader.cleanup_partial_load')
	def test_cleans_up_on_loading_error(
		self,
		mock_cleanup: Mock,
		mock_device_service: Mock,
		mock_socket_service: Mock,
		mock_auto_pipeline: Mock,
		mock_storage_service: Mock,
		mock_safety_checker: Mock,
		mock_feature_extractor: Mock,
		mock_max_memory: Mock,
		mock_session_local: Mock,
	) -> None:
		"""Test that partial load is cleaned up when loading fails."""
		mock_db = Mock()
		mock_session_local.return_value = mock_db
		mock_max_memory.return_value.to_dict.return_value = {}
		mock_device_service.device = 'cuda'
		mock_storage_service.get_model_dir.return_value = '/cache/model'

		# Make loading fail
		mock_auto_pipeline.from_pretrained.side_effect = RuntimeError('Load failed')

		with pytest.raises(RuntimeError):
			model_loader('test-model')

		mock_cleanup.assert_called_once()
		mock_db.close.assert_called_once()

	@patch('app.cores.model_loader.model_loader.SessionLocal')
	@patch('app.cores.model_loader.model_loader.MaxMemoryConfig')
	@patch('app.cores.model_loader.model_loader.CLIPImageProcessor')
	@patch('app.cores.model_loader.model_loader.StableDiffusionSafetyChecker')
	@patch('app.cores.model_loader.model_loader.storage_service')
	def test_raises_cancellation_at_various_checkpoints(
		self,
		mock_storage_service: Mock,
		mock_safety_checker: Mock,
		mock_feature_extractor: Mock,
		mock_max_memory: Mock,
		mock_session_local: Mock,
	) -> None:
		"""Test that cancellation is checked at multiple checkpoints."""
		mock_db = Mock()
		mock_session_local.return_value = mock_db
		mock_max_memory.return_value.to_dict.return_value = {}
		mock_storage_service.get_model_dir.return_value = '/cache/model'

		# Test cancellation at checkpoint 4
		cancel_token = CancellationToken()
		call_count = {'count': 0}

		original_check = cancel_token.check_cancelled

		def delayed_cancel():
			call_count['count'] += 1
			if call_count['count'] >= 4:
				cancel_token.cancel()
			original_check()

		setattr(cancel_token, 'check_cancelled', delayed_cancel)

		with pytest.raises(CancellationException):
			model_loader('test-model', cancel_token=cancel_token)

		mock_db.close.assert_called_once()

	@patch('app.cores.model_loader.model_loader.SessionLocal')
	@patch('app.cores.model_loader.model_loader.MaxMemoryConfig')
	@patch('app.cores.model_loader.model_loader.CLIPImageProcessor')
	@patch('app.cores.model_loader.model_loader.StableDiffusionSafetyChecker')
	@patch('app.cores.model_loader.model_loader.storage_service')
	@patch('app.cores.model_loader.model_loader.AutoPipelineForText2Image')
	@patch('app.cores.model_loader.model_loader.socket_service')
	@patch('app.cores.model_loader.model_loader.device_service')
	@patch('app.cores.model_loader.model_loader.apply_device_optimizations')
	@patch('app.cores.model_loader.model_loader.move_to_device')
	def test_resets_device_map_when_available(
		self,
		mock_move_to_device: Mock,
		mock_apply_optimizations: Mock,
		mock_device_service: Mock,
		mock_socket_service: Mock,
		mock_auto_pipeline: Mock,
		mock_storage_service: Mock,
		mock_safety_checker: Mock,
		mock_feature_extractor: Mock,
		mock_max_memory: Mock,
		mock_session_local: Mock,
	) -> None:
		"""Test that device map is reset when pipeline has the method."""
		mock_db = Mock()
		mock_session_local.return_value = mock_db
		mock_max_memory.return_value.to_dict.return_value = {}
		mock_device_service.device = 'cuda'
		mock_device_service.torch_dtype = 'float16'
		mock_storage_service.get_model_dir.return_value = '/cache/model'

		mock_pipe = Mock()
		mock_pipe.reset_device_map = Mock()
		mock_auto_pipeline.from_pretrained.return_value = mock_pipe
		mock_move_to_device.return_value = mock_pipe

		model_loader('test-model')

		mock_pipe.reset_device_map.assert_called_once()

	@patch('app.cores.model_loader.model_loader.SessionLocal')
	@patch('app.cores.model_loader.model_loader.MaxMemoryConfig')
	@patch('app.cores.model_loader.model_loader.CLIPImageProcessor')
	@patch('app.cores.model_loader.model_loader.StableDiffusionSafetyChecker')
	@patch('app.cores.model_loader.model_loader.storage_service')
	@patch('app.cores.model_loader.model_loader.AutoPipelineForText2Image')
	@patch('app.cores.model_loader.model_loader.socket_service')
	@patch('app.cores.model_loader.model_loader.device_service')
	@patch('app.cores.model_loader.model_loader.apply_device_optimizations')
	@patch('app.cores.model_loader.model_loader.move_to_device')
	def test_handles_pipeline_without_reset_device_map(
		self,
		mock_move_to_device: Mock,
		mock_apply_optimizations: Mock,
		mock_device_service: Mock,
		mock_socket_service: Mock,
		mock_auto_pipeline: Mock,
		mock_storage_service: Mock,
		mock_safety_checker: Mock,
		mock_feature_extractor: Mock,
		mock_max_memory: Mock,
		mock_session_local: Mock,
	) -> None:
		"""Test that loading works when pipeline doesn't have reset_device_map."""
		mock_db = Mock()
		mock_session_local.return_value = mock_db
		mock_max_memory.return_value.to_dict.return_value = {}
		mock_device_service.device = 'cuda'
		mock_device_service.torch_dtype = 'float16'
		mock_storage_service.get_model_dir.return_value = '/cache/model'

		# Pipeline without reset_device_map method
		mock_pipe = Mock(spec=['to', 'to_empty'])
		if hasattr(mock_pipe, 'reset_device_map'):
			delattr(mock_pipe, 'reset_device_map')

		mock_auto_pipeline.from_pretrained.return_value = mock_pipe
		mock_move_to_device.return_value = mock_pipe

		result = model_loader('test-model')

		assert result is mock_pipe

	@patch('app.cores.model_loader.model_loader.SessionLocal')
	@patch('app.cores.model_loader.model_loader.MaxMemoryConfig')
	@patch('app.cores.model_loader.model_loader.CLIPImageProcessor')
	@patch('app.cores.model_loader.model_loader.StableDiffusionSafetyChecker')
	@patch('app.cores.model_loader.model_loader.storage_service')
	@patch('app.cores.model_loader.model_loader.AutoPipelineForText2Image')
	@patch('app.cores.model_loader.model_loader.socket_service')
	@patch('app.cores.model_loader.model_loader.device_service')
	@patch('app.cores.model_loader.model_loader.apply_device_optimizations')
	@patch('app.cores.model_loader.model_loader.move_to_device')
	def test_cancellation_at_checkpoint_6(
		self,
		mock_move_to_device: Mock,
		mock_apply_optimizations: Mock,
		mock_device_service: Mock,
		mock_socket_service: Mock,
		mock_auto_pipeline: Mock,
		mock_storage_service: Mock,
		mock_safety_checker: Mock,
		mock_feature_extractor: Mock,
		mock_max_memory: Mock,
		mock_session_local: Mock,
	) -> None:
		"""Test cancellation at checkpoint 6 (after pipeline loaded)."""
		mock_db = Mock()
		mock_session_local.return_value = mock_db
		mock_max_memory.return_value.to_dict.return_value = {}
		mock_device_service.device = 'cuda'
		mock_device_service.torch_dtype = 'float16'
		mock_storage_service.get_model_dir.return_value = '/cache/model'

		mock_pipe = Mock()
		mock_auto_pipeline.from_pretrained.return_value = mock_pipe

		cancel_token = CancellationToken()
		call_count = {'count': 0}

		original_check = cancel_token.check_cancelled

		def delayed_cancel():
			call_count['count'] += 1
			if call_count['count'] >= 6:
				cancel_token.cancel()
			original_check()

		setattr(cancel_token, 'check_cancelled', delayed_cancel)

		with pytest.raises(CancellationException):
			model_loader('test-model', cancel_token=cancel_token)

		mock_db.close.assert_called_once()

	@patch('app.cores.model_loader.model_loader.SessionLocal')
	@patch('app.cores.model_loader.model_loader.MaxMemoryConfig')
	@patch('app.cores.model_loader.model_loader.CLIPImageProcessor')
	@patch('app.cores.model_loader.model_loader.StableDiffusionSafetyChecker')
	@patch('app.cores.model_loader.model_loader.storage_service')
	@patch('app.cores.model_loader.model_loader.AutoPipelineForText2Image')
	@patch('app.cores.model_loader.model_loader.socket_service')
	@patch('app.cores.model_loader.model_loader.device_service')
	@patch('app.cores.model_loader.model_loader.apply_device_optimizations')
	@patch('app.cores.model_loader.model_loader.move_to_device')
	def test_cancellation_at_checkpoint_8(
		self,
		mock_move_to_device: Mock,
		mock_apply_optimizations: Mock,
		mock_device_service: Mock,
		mock_socket_service: Mock,
		mock_auto_pipeline: Mock,
		mock_storage_service: Mock,
		mock_safety_checker: Mock,
		mock_feature_extractor: Mock,
		mock_max_memory: Mock,
		mock_session_local: Mock,
	) -> None:
		"""Test cancellation at checkpoint 8 (after device operations)."""
		mock_db = Mock()
		mock_session_local.return_value = mock_db
		mock_max_memory.return_value.to_dict.return_value = {}
		mock_device_service.device = 'cuda'
		mock_device_service.torch_dtype = 'float16'
		mock_storage_service.get_model_dir.return_value = '/cache/model'

		mock_pipe = Mock()
		mock_auto_pipeline.from_pretrained.return_value = mock_pipe
		mock_move_to_device.return_value = mock_pipe

		cancel_token = CancellationToken()
		call_count = {'count': 0}

		original_check = cancel_token.check_cancelled

		def delayed_cancel():
			call_count['count'] += 1
			if call_count['count'] >= 8:
				cancel_token.cancel()
			original_check()

		setattr(cancel_token, 'check_cancelled', delayed_cancel)

		with pytest.raises(CancellationException):
			model_loader('test-model', cancel_token=cancel_token)

		mock_db.close.assert_called_once()

	@patch('app.cores.model_loader.model_loader.SessionLocal')
	@patch('app.cores.model_loader.model_loader.MaxMemoryConfig')
	@patch('app.cores.model_loader.model_loader.CLIPImageProcessor')
	@patch('app.cores.model_loader.model_loader.StableDiffusionSafetyChecker')
	@patch('app.cores.model_loader.model_loader.storage_service')
	@patch('app.cores.model_loader.model_loader.AutoPipelineForText2Image')
	@patch('app.cores.model_loader.model_loader.socket_service')
	@patch('app.cores.model_loader.model_loader.device_service')
	@patch('app.cores.model_loader.model_loader.apply_device_optimizations')
	@patch('app.cores.model_loader.model_loader.move_to_device')
	def test_cancellation_at_checkpoint_9(
		self,
		mock_move_to_device: Mock,
		mock_apply_optimizations: Mock,
		mock_device_service: Mock,
		mock_socket_service: Mock,
		mock_auto_pipeline: Mock,
		mock_storage_service: Mock,
		mock_safety_checker: Mock,
		mock_feature_extractor: Mock,
		mock_max_memory: Mock,
		mock_session_local: Mock,
	) -> None:
		"""Test cancellation at checkpoint 9 (final checkpoint)."""
		mock_db = Mock()
		mock_session_local.return_value = mock_db
		mock_max_memory.return_value.to_dict.return_value = {}
		mock_device_service.device = 'cuda'
		mock_device_service.torch_dtype = 'float16'
		mock_storage_service.get_model_dir.return_value = '/cache/model'

		mock_pipe = Mock()
		mock_auto_pipeline.from_pretrained.return_value = mock_pipe
		mock_move_to_device.return_value = mock_pipe

		cancel_token = CancellationToken()
		call_count = {'count': 0}

		original_check = cancel_token.check_cancelled

		def delayed_cancel():
			call_count['count'] += 1
			if call_count['count'] >= 9:
				cancel_token.cancel()
			original_check()

		setattr(cancel_token, 'check_cancelled', delayed_cancel)

		with pytest.raises(CancellationException):
			model_loader('test-model', cancel_token=cancel_token)

		mock_db.close.assert_called_once()

	@patch('app.cores.model_loader.model_loader.SessionLocal')
	@patch('app.cores.model_loader.model_loader.MaxMemoryConfig')
	@patch('app.cores.model_loader.model_loader.CLIPImageProcessor')
	@patch('app.cores.model_loader.model_loader.StableDiffusionSafetyChecker')
	@patch('app.cores.model_loader.model_loader.storage_service')
	@patch('app.cores.model_loader.model_loader.socket_service')
	@patch('app.cores.model_loader.model_loader.device_service')
	@patch('app.cores.model_loader.model_loader.cleanup_partial_load')
	def test_raises_runtime_error_when_no_error_captured(
		self,
		mock_cleanup: Mock,
		mock_device_service: Mock,
		mock_socket_service: Mock,
		mock_storage_service: Mock,
		mock_safety_checker: Mock,
		mock_feature_extractor: Mock,
		mock_max_memory: Mock,
		mock_session_local: Mock,
	) -> None:
		"""Test RuntimeError when all strategies fail but no error is captured."""
		mock_db = Mock()
		mock_session_local.return_value = mock_db
		mock_max_memory.return_value.to_dict.return_value = {}
		mock_device_service.device = 'cuda'
		mock_storage_service.get_model_dir.return_value = '/cache/model'

		# Patch AutoPipelineForText2Image to return None without raising
		with patch('app.cores.model_loader.model_loader.AutoPipelineForText2Image') as mock_auto:
			mock_auto.from_pretrained.return_value = None

			with pytest.raises(RuntimeError, match='Failed to load model test-model'):
				model_loader('test-model')

		mock_cleanup.assert_called_once()
		mock_db.close.assert_called_once()
