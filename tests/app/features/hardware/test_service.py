"""Tests for hardware service."""

from unittest.mock import MagicMock, patch

from app.features.hardware.info import GPUInfo
from app.features.hardware.service import HardwareService
from app.schemas.hardware import (
	GetCurrentDeviceIndex,
	GPUDriverInfo,
	GPUDriverStatusStates,
	MaxMemoryConfigRequest,
	MemoryResponse,
)


class TestHardwareService:
	"""Test hardware service orchestration."""

	def setup_method(self):
		"""Set up test fixtures."""
		self.service = HardwareService()

	def test_init_creates_gpu_detector(self):
		"""Test __init__ creates GPU detector."""
		assert self.service.gpu_detector is not None

	def test_get_gpu_info_calls_detector(self):
		"""Test get_gpu_info() calls GPU detector."""
		mock_info = GPUDriverInfo(
			gpus=[],
			is_cuda=True,
			message=GPUInfo.nvidia_ready(),
			overall_status=GPUDriverStatusStates.READY,
		)

		with patch.object(self.service.gpu_detector, 'detect', return_value=mock_info) as mock_detect:
			result = self.service.get_gpu_info()

			assert result == mock_info
			mock_detect.assert_called_once()

	def test_recheck_gpu_info_clears_cache_and_detects(self):
		"""Test recheck_gpu_info() clears cache before detecting."""
		mock_info = GPUDriverInfo(
			gpus=[],
			is_cuda=True,
			message=GPUInfo.nvidia_ready(),
			overall_status=GPUDriverStatusStates.READY,
		)

		with patch.object(self.service.gpu_detector, 'clear_cache') as mock_clear:
			with patch.object(self.service.gpu_detector, 'detect', return_value=mock_info) as mock_detect:
				result = self.service.recheck_gpu_info()

				mock_clear.assert_called_once()
				mock_detect.assert_called_once()
				assert result == mock_info

	@patch('app.features.hardware.service.MemoryService')
	def test_get_memory_info_returns_memory_response(self, mock_memory_service_class):
		"""Test get_memory_info() returns MemoryResponse."""
		mock_db = MagicMock()
		mock_memory_service = MagicMock()
		mock_memory_service.total_gpu = 8192
		mock_memory_service.total_ram = 16384
		mock_memory_service_class.return_value = mock_memory_service

		result = self.service.get_memory_info(mock_db)

		assert isinstance(result, MemoryResponse)
		assert result.gpu == 8192
		assert result.ram == 16384
		mock_memory_service_class.assert_called_once_with(mock_db)

	@patch('app.features.hardware.service.add_device_index')
	def test_set_device_calls_add_device_index(self, mock_add_device_index):
		"""Test set_device() calls add_device_index."""
		mock_db = MagicMock()
		device_index = 1

		result = self.service.set_device(mock_db, device_index)

		mock_add_device_index.assert_called_once_with(mock_db, device_index=device_index)
		assert result['device_index'] == device_index

	@patch('app.features.hardware.service.add_device_index')
	def test_set_device_returns_success_message(self, mock_add_device_index):
		"""Test set_device() returns success message."""
		mock_db = MagicMock()
		device_index = 1

		result = self.service.set_device(mock_db, device_index)

		assert result['message'] == GPUInfo.device_set_success()

	@patch('app.features.hardware.service.get_device_index')
	def test_get_device_returns_device_index(self, mock_get_device_index):
		"""Test get_device() returns device index."""
		mock_db = MagicMock()
		mock_get_device_index.return_value = 0

		result = self.service.get_device(mock_db)

		assert isinstance(result, GetCurrentDeviceIndex)
		assert result.device_index == 0
		mock_get_device_index.assert_called_once_with(mock_db)

	@patch('app.features.hardware.service.add_max_memory')
	def test_set_max_memory_calls_add_max_memory(self, mock_add_max_memory):
		"""Test set_max_memory() calls add_max_memory."""
		mock_db = MagicMock()
		config = MaxMemoryConfigRequest(ram_scale_factor=0.8, gpu_scale_factor=0.9)

		result = self.service.set_max_memory(mock_db, config)

		mock_add_max_memory.assert_called_once_with(mock_db, ram_scale_factor=0.8, gpu_scale_factor=0.9)
		assert abs(result['ram_scale_factor'] - 0.8) < 1e-9
		assert abs(result['gpu_scale_factor'] - 0.9) < 1e-9

	@patch('app.features.hardware.service.add_max_memory')
	def test_set_max_memory_returns_success_message(self, mock_add_max_memory):
		"""Test set_max_memory() returns success message."""
		mock_db = MagicMock()
		config = MaxMemoryConfigRequest(ram_scale_factor=0.8, gpu_scale_factor=0.9)

		result = self.service.set_max_memory(mock_db, config)

		assert result['message'] == GPUInfo.memory_config_success()
