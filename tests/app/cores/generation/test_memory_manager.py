"""Tests for memory_manager module."""

from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def mock_memory_manager():
	"""Create MemoryManager with mocked dependencies."""
	with (
		patch('app.cores.generation.memory_manager.device_service') as mock_device_service,
		patch('app.cores.generation.memory_manager.torch') as mock_torch,
	):
		# Configure device_service
		mock_device_service.is_cuda = False
		mock_device_service.is_mps = False
		mock_device_service.get_recommended_batch_size.return_value = 3

		# Configure torch
		mock_torch.cuda.empty_cache = Mock()
		mock_torch.mps.empty_cache = Mock()

		from app.cores.generation.memory_manager import MemoryManager

		manager = MemoryManager()

		yield manager, mock_device_service, mock_torch


class TestClearCache:
	def test_clears_cuda_cache_when_cuda_available(self, mock_memory_manager):
		manager, mock_device_service, mock_torch = mock_memory_manager
		mock_device_service.is_cuda = True

		manager.clear_cache()

		mock_torch.cuda.empty_cache.assert_called_once()

	def test_clears_mps_cache_when_mps_available(self, mock_memory_manager):
		manager, mock_device_service, mock_torch = mock_memory_manager
		mock_device_service.is_mps = True

		manager.clear_cache()

		mock_torch.mps.empty_cache.assert_called_once()

	def test_does_nothing_when_no_accelerator(self, mock_memory_manager):
		manager, mock_device_service, mock_torch = mock_memory_manager
		mock_device_service.is_cuda = False
		mock_device_service.is_mps = False

		manager.clear_cache()

		mock_torch.cuda.empty_cache.assert_not_called()
		mock_torch.mps.empty_cache.assert_not_called()


class TestValidateBatchSize:
	def test_logs_warning_when_batch_exceeds_recommended(self, mock_memory_manager, caplog):
		manager, mock_device_service, _ = mock_memory_manager
		mock_device_service.get_recommended_batch_size.return_value = 3

		manager.validate_batch_size(number_of_images=5, width=512, height=512)

		assert 'may cause OOM errors' in caplog.text
		assert 'Recommended: 3' in caplog.text

	def test_no_warning_when_batch_within_recommended(self, mock_memory_manager, caplog):
		manager, mock_device_service, _ = mock_memory_manager
		mock_device_service.get_recommended_batch_size.return_value = 3

		manager.validate_batch_size(number_of_images=2, width=512, height=512)

		assert 'may cause OOM errors' not in caplog.text
