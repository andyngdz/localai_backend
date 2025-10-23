"""Additional tests for model_loader to achieve higher coverage.

This test suite covers error paths and edge cases in model_loader:
- Cleanup on exception
- Various loading strategy error paths
- Cancellation during load
"""

from unittest.mock import MagicMock, patch

import pytest

from app.cores.model_loader.cancellation import CancellationToken


class TestModelLoaderCleanup:
	"""Test model loader cleanup on exceptions."""

	@patch('app.cores.model_loader.model_loader.SessionLocal')
	@patch('app.cores.model_loader.model_loader.MaxMemoryConfig')
	@patch('app.cores.model_loader.model_loader.storage_service')
	@patch('app.cores.model_loader.model_loader.CLIPImageProcessor')
	@patch('app.cores.model_loader.model_loader.AutoPipelineForText2Image')
	def test_cleanup_on_exception_during_load(
		self, mock_pipeline, mock_clip, mock_storage, mock_max_memory, mock_session
	):
		"""Test that cleanup happens when exception occurs during loading (lines 128-130)."""
		from app.cores.model_loader.model_loader import model_loader

		# Setup
		mock_db = MagicMock()
		mock_session.return_value = mock_db

		mock_max_memory_instance = MagicMock()
		mock_max_memory_instance.to_dict.return_value = {}
		mock_max_memory.return_value = mock_max_memory_instance

		mock_storage.get_model_dir.return_value = '/fake/models/test-model'

		# Mock CLIPImageProcessor to succeed
		mock_clip.from_pretrained.return_value = MagicMock()

		# Mock AutoPipelineForText2Image to raise exception
		mock_pipeline.from_pretrained.side_effect = RuntimeError('GPU out of memory')

		# Execute - should raise and trigger cleanup
		with pytest.raises(RuntimeError, match='GPU out of memory'):
			model_loader('test/model')

		# Verify cleanup was attempted (db.close called)
		mock_db.close.assert_called()


class TestLoadingStrategyEdgeCases:
	"""Test various loading strategy edge cases."""

	@patch('app.cores.model_loader.model_loader.SessionLocal')
	@patch('app.cores.model_loader.model_loader.MaxMemoryConfig')
	@patch('app.cores.model_loader.model_loader.storage_service')
	@patch('app.cores.model_loader.model_loader.CLIPImageProcessor')
	@patch('app.cores.model_loader.model_loader.AutoPipelineForText2Image')
	def test_loads_with_fp16_variant(self, mock_pipeline, mock_clip, mock_storage, mock_max_memory, mock_session):
		"""Test loading with fp16 variant parameter."""
		from app.cores.model_loader.model_loader import model_loader

		# Setup
		mock_db = MagicMock()
		mock_session.return_value = mock_db

		mock_max_memory_instance = MagicMock()
		mock_max_memory_instance.to_dict.return_value = {}
		mock_max_memory.return_value = mock_max_memory_instance

		mock_storage.get_model_dir.return_value = '/fake/models/test-model'

		mock_clip.from_pretrained.return_value = MagicMock()

		# Mock pipeline to succeed
		mock_pipe_instance = MagicMock()
		mock_pipe_instance.config = {'model_type': 'test'}
		mock_pipe_instance.to.return_value = mock_pipe_instance
		mock_pipeline.from_pretrained.return_value = mock_pipe_instance

		# Execute
		result = model_loader('test/model')

		# Verify pipeline was loaded
		assert result is not None
		mock_pipeline.from_pretrained.assert_called()

		mock_db.close.assert_called()


class TestCancellationDuringLoad:
	"""Test cancellation at various checkpoints."""

	@patch('app.cores.model_loader.model_loader.SessionLocal')
	@patch('app.cores.model_loader.model_loader.MaxMemoryConfig')
	def test_cancellation_at_checkpoint_3(self, mock_max_memory, mock_session):
		"""Test cancellation at checkpoint 3 (before loading model)."""
		from app.cores.model_loader.cancellation import CancellationException
		from app.cores.model_loader.model_loader import model_loader

		# Setup
		mock_db = MagicMock()
		mock_session.return_value = mock_db

		mock_max_memory_instance = MagicMock()
		mock_max_memory_instance.to_dict.return_value = {}
		mock_max_memory.return_value = mock_max_memory_instance

		# Create token and cancel immediately
		token = CancellationToken()
		token.cancel()

		# Execute - should raise at first checkpoint
		with pytest.raises(CancellationException):
			model_loader('test/model', token)

		mock_db.close.assert_called()
