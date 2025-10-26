"""Tests for gpu_utils module."""

from unittest.mock import MagicMock, patch


class TestCleanupGpuModel:
	"""Test cleanup_gpu_model function."""

	def test_cleanup_with_none_model_returns_zero_metrics(self):
		"""Test that cleanup with None model returns zero metrics (line 32)."""
		from app.cores.gpu_utils import cleanup_gpu_model

		# Execute
		result = cleanup_gpu_model(None, 'test_model')

		# Verify
		assert result.time_ms == 0
		assert result.objects_collected == 0
		assert result.error is None

	@patch('app.cores.gpu_utils.gc')
	@patch('app.cores.gpu_utils.torch')
	def test_cleanup_with_cuda_available(self, mock_torch, mock_gc):
		"""Test cleanup with CUDA available."""
		from app.cores.gpu_utils import cleanup_gpu_model

		# Setup
		mock_model = MagicMock()
		mock_gc.collect.return_value = 42
		mock_torch.cuda.is_available.return_value = True

		# Execute
		result = cleanup_gpu_model(mock_model, 'test_model')

		# Verify
		assert result.time_ms > 0
		assert result.objects_collected == 42
		assert result.error is None
		mock_gc.collect.assert_called_once()
		mock_torch.cuda.empty_cache.assert_called_once()

	@patch('app.cores.gpu_utils.gc')
	@patch('app.cores.gpu_utils.torch')
	def test_cleanup_without_cuda(self, mock_torch, mock_gc):
		"""Test cleanup without CUDA available."""
		from app.cores.gpu_utils import cleanup_gpu_model

		# Setup
		mock_model = MagicMock()
		mock_gc.collect.return_value = 15
		mock_torch.cuda.is_available.return_value = False

		# Execute
		result = cleanup_gpu_model(mock_model, 'test_model')

		# Verify
		assert result.time_ms > 0
		assert result.objects_collected == 15
		assert result.error is None
		mock_gc.collect.assert_called_once()
		mock_torch.cuda.empty_cache.assert_not_called()

	@patch('app.cores.gpu_utils.gc')
	@patch('app.cores.gpu_utils.torch')
	def test_cleanup_handles_exception(self, mock_torch, mock_gc):
		"""Test cleanup handles exceptions gracefully (lines 48-50)."""
		from app.cores.gpu_utils import cleanup_gpu_model

		# Setup - make gc.collect raise an exception
		mock_model = MagicMock()
		mock_gc.collect.side_effect = RuntimeError('GC failed')

		# Execute - should not raise
		result = cleanup_gpu_model(mock_model, 'test_model')

		# Verify
		assert result.time_ms == 0
		assert result.objects_collected == 0
		assert result.error == 'GC failed'

	@patch('app.cores.gpu_utils.time')
	@patch('app.cores.gpu_utils.gc')
	@patch('app.cores.gpu_utils.torch')
	def test_cleanup_measures_time_correctly(self, mock_torch, mock_gc, mock_time):
		"""Test that cleanup measures elapsed time correctly."""
		from app.cores.gpu_utils import cleanup_gpu_model

		# Setup - mock time to return specific values
		mock_time.time.side_effect = [1.0, 1.5]  # 500ms elapsed
		mock_model = MagicMock()
		mock_gc.collect.return_value = 10
		mock_torch.cuda.is_available.return_value = False

		# Execute
		result = cleanup_gpu_model(mock_model, 'test_model')

		# Verify - 500ms * 1000 = 500 milliseconds
		assert result.time_ms == 500.0
		assert result.objects_collected == 10
