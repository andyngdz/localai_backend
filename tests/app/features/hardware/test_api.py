"""Integration tests for hardware API endpoints and detection logic."""

from unittest.mock import MagicMock, patch

import pytest

from app.features.hardware.info import GPUInfo
from app.schemas.hardware import GPUDriverInfo, GPUDriverStatusStates


class TestHardwareEndpoints:
	"""Test hardware API endpoints."""

	@patch('app.features.hardware.service.hardware_service.get_gpu_info')
	def test_get_hardware_endpoint(self, mock_get_gpu_info):
		"""Test GET /hardware/ endpoint returns GPU info."""
		from app.features.hardware.api import get_hardware

		mock_info = GPUDriverInfo(
			gpus=[],
			is_cuda=False,
			message=GPUInfo.default_detecting(),
			overall_status=GPUDriverStatusStates.UNKNOWN_ERROR,
		)
		mock_get_gpu_info.return_value = mock_info

		result = get_hardware()

		assert result == mock_info
		mock_get_gpu_info.assert_called_once()

	@patch('app.features.hardware.service.hardware_service.recheck_gpu_info')
	def test_recheck_endpoint(self, mock_recheck_gpu_info):
		"""Test GET /hardware/recheck re-detects GPU."""
		from app.features.hardware.api import recheck

		mock_info = GPUDriverInfo(
			gpus=[],
			is_cuda=True,
			message=GPUInfo.nvidia_ready(),
			overall_status=GPUDriverStatusStates.READY,
		)
		mock_recheck_gpu_info.return_value = mock_info

		result = recheck()

		assert result == mock_info
		mock_recheck_gpu_info.assert_called_once()

	@patch('app.features.hardware.service.hardware_service.set_device')
	def test_set_device_endpoint(self, mock_set_device):
		"""Test POST /hardware/device sets device index."""
		from app.features.hardware.api import set_device
		from app.schemas.hardware import SelectDeviceRequest

		mock_db = MagicMock()
		request = SelectDeviceRequest(device_index=1)
		mock_set_device.return_value = {'message': GPUInfo.device_set_success(), 'device_index': 1}

		result = set_device(request, db=mock_db)

		assert result['message'] == GPUInfo.device_set_success()
		assert result['device_index'] == 1
		mock_set_device.assert_called_once_with(mock_db, 1)

	@patch('app.features.hardware.service.hardware_service.get_device')
	def test_get_device_endpoint(self, mock_get_device):
		"""Test GET /hardware/device returns current device index."""
		from app.features.hardware.api import get_device
		from app.schemas.hardware import GetCurrentDeviceIndex

		mock_db = MagicMock()
		mock_get_device.return_value = GetCurrentDeviceIndex(device_index=0)

		result = get_device(db=mock_db)

		assert result.device_index == 0
		mock_get_device.assert_called_once_with(mock_db)

	@patch('app.features.hardware.service.hardware_service.get_device')
	def test_get_device_endpoint_error_handling(self, mock_get_device):
		"""Test GET /hardware/device handles errors."""
		from fastapi import HTTPException

		from app.features.hardware.api import get_device

		mock_db = MagicMock()
		error_msg = 'Database error'
		mock_get_device.side_effect = Exception(error_msg)

		with pytest.raises(HTTPException) as exc_info:
			get_device(db=mock_db)

		assert exc_info.value.status_code == 500
		assert error_msg in str(exc_info.value.detail)

	@patch('app.features.hardware.service.hardware_service.set_max_memory')
	def test_set_max_memory_endpoint(self, mock_set_max_memory):
		"""Test POST /hardware/max-memory sets memory configuration."""
		from app.features.hardware.api import set_max_memory
		from app.schemas.hardware import MaxMemoryConfigRequest

		mock_db = MagicMock()
		config = MaxMemoryConfigRequest(ram_scale_factor=0.8, gpu_scale_factor=0.9)
		mock_set_max_memory.return_value = {
			'message': GPUInfo.memory_config_success(),
			'ram_scale_factor': 0.8,
			'gpu_scale_factor': 0.9,
		}

		result = set_max_memory(config, db=mock_db)

		assert result['message'] == GPUInfo.memory_config_success()
		assert result['ram_scale_factor'] == 0.8
		assert result['gpu_scale_factor'] == 0.9
		mock_set_max_memory.assert_called_once_with(mock_db, config)

	@patch('app.features.hardware.service.hardware_service.get_memory_info')
	def test_get_device_memory_endpoint(self, mock_get_memory_info):
		"""Test GET /hardware/memory returns memory info."""
		from app.features.hardware.api import get_device_memory
		from app.schemas.hardware import MemoryResponse

		mock_db = MagicMock()
		mock_get_memory_info.return_value = MemoryResponse(gpu=8192, ram=16384)

		result = get_device_memory(db=mock_db)

		assert result.gpu == 8192
		assert result.ram == 16384
		mock_get_memory_info.assert_called_once_with(mock_db)

	@patch('app.features.hardware.service.hardware_service.get_memory_info')
	def test_get_device_memory_endpoint_error_handling(self, mock_get_memory_info):
		"""Test GET /hardware/memory handles errors."""
		from fastapi import HTTPException

		from app.features.hardware.api import get_device_memory

		mock_db = MagicMock()
		error_msg = 'Memory service error'
		mock_get_memory_info.side_effect = Exception(error_msg)

		with pytest.raises(HTTPException) as exc_info:
			get_device_memory(db=mock_db)

		assert exc_info.value.status_code == 500
		assert error_msg in str(exc_info.value.detail)
