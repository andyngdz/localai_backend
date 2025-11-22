"""Integration tests for hardware API endpoints and detection logic."""

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from app.features.hardware.api import (
	default_gpu_info,
	get_mps_gpu_info,
	get_nvidia_gpu_info,
	get_system_gpu_info,
)
from app.features.hardware.info import GPUInfo
from app.schemas.hardware import GPUDriverInfo, GPUDriverStatusStates


class TestDefaultGPUInfo:
	"""Test default GPU info initialization."""

	@patch('app.features.hardware.api.device_service')
	def test_default_gpu_info_initialization(self, mock_device_service):
		"""Test default_gpu_info initializes with correct defaults."""
		mock_device_service.is_cuda = False

		info = default_gpu_info()

		assert isinstance(info, GPUDriverInfo)
		assert info.overall_status == GPUDriverStatusStates.UNKNOWN_ERROR
		assert info.message == GPUInfo.default_detecting()
		assert info.gpus == []
		assert info.is_cuda is False


class TestGetNvidiaGPUInfo:
	"""Test NVIDIA GPU detection logic."""

	def setup_method(self):
		"""Set up test fixtures."""
		self.info = GPUDriverInfo(
			gpus=[],
			is_cuda=False,
			message='',
			overall_status=GPUDriverStatusStates.UNKNOWN_ERROR,
		)

	@patch('app.features.hardware.api.device_service')
	@patch('app.features.hardware.api.subprocess')
	@patch('app.features.hardware.api.torch')
	def test_nvidia_gpu_ready_with_cuda(self, mock_torch, mock_subprocess, mock_device_service):
		"""Test NVIDIA GPU detected with CUDA available."""
		# Setup mocks
		mock_device_service.is_cuda = True
		mock_device_service.device_count = 1
		mock_device_service.get_device_name.return_value = 'NVIDIA GeForce RTX 3090'
		mock_device_service.current_device = 0

		mock_device_property = MagicMock()
		mock_device_property.total_memory = 24 * 1024 * 1024 * 1024  # 24GB
		mock_device_property.major = 8
		mock_device_property.minor = 6
		mock_device_service.get_device_properties.return_value = mock_device_property

		mock_version = MagicMock()
		mock_version.cuda = '11.8'
		mock_torch.version = mock_version

		mock_result = MagicMock()
		mock_result.stdout = '525.60.11\n'
		mock_subprocess.run.return_value = mock_result

		# Execute
		get_nvidia_gpu_info('Linux', self.info)

		# Assert
		assert self.info.overall_status == GPUDriverStatusStates.READY
		assert self.info.message == GPUInfo.nvidia_ready()
		assert self.info.cuda_runtime_version == '11.8'
		assert self.info.nvidia_driver_version == '525.60.11'
		assert len(self.info.gpus) == 1
		assert self.info.gpus[0].name == 'NVIDIA GeForce RTX 3090'

	@patch('app.features.hardware.api.subprocess.run')
	@patch('app.features.hardware.api.device_service')
	def test_nvidia_no_cuda_on_linux_shows_nvidia_troubleshooting(self, mock_device_service, mock_subprocess_run):
		"""Test no CUDA on Linux shows NVIDIA troubleshooting steps."""
		mock_device_service.is_cuda = False
		mock_subprocess_run.side_effect = FileNotFoundError()

		get_nvidia_gpu_info('Linux', self.info)

		assert self.info.overall_status == GPUDriverStatusStates.NO_GPU
		assert self.info.message == GPUInfo.nvidia_no_gpu()
		assert self.info.recommendation_link == GPUInfo.nvidia_recommendation_link()
		assert self.info.troubleshooting_steps == GPUInfo.nvidia_troubleshooting_steps()

	@patch('app.features.hardware.api.subprocess.run')
	@patch('app.features.hardware.api.device_service')
	def test_nvidia_no_cuda_on_windows_shows_nvidia_troubleshooting(self, mock_device_service, mock_subprocess_run):
		"""Test no CUDA on Windows shows NVIDIA troubleshooting steps."""
		mock_device_service.is_cuda = False
		mock_subprocess_run.side_effect = FileNotFoundError()

		get_nvidia_gpu_info('Windows', self.info)

		assert self.info.overall_status == GPUDriverStatusStates.NO_GPU
		assert self.info.message == GPUInfo.nvidia_no_gpu()
		assert self.info.recommendation_link == GPUInfo.nvidia_recommendation_link()
		assert self.info.troubleshooting_steps == GPUInfo.nvidia_troubleshooting_steps()

	@patch('app.features.hardware.api.device_service')
	def test_nvidia_no_cuda_on_macos_shows_macos_message_no_nvidia_troubleshooting(self, mock_device_service):
		"""Test no CUDA on macOS shows macOS message without NVIDIA troubleshooting."""
		mock_device_service.is_cuda = False

		get_nvidia_gpu_info('Darwin', self.info)

		assert self.info.overall_status == GPUDriverStatusStates.NO_GPU
		assert self.info.message == GPUInfo.macos_no_acceleration()
		# Critical: macOS should NOT have NVIDIA recommendations or troubleshooting
		# Schema defaults are '' for link and [] for steps, so check they weren't changed
		assert self.info.recommendation_link == '', 'macOS should have empty recommendation link (not NVIDIA)'
		assert self.info.troubleshooting_steps == [], 'macOS should have no troubleshooting steps (not NVIDIA)'

	@patch('app.features.hardware.api.subprocess.run')
	@patch('app.features.hardware.api.torch')
	@patch('app.features.hardware.api.device_service')
	def test_nvidia_smi_failure_appends_warning(self, mock_device_service, mock_torch, mock_subprocess_run):
		"""Test nvidia-smi failure appends warning to message."""
		mock_device_service.is_cuda = True
		mock_device_service.device_count = 0

		mock_version = MagicMock()
		mock_version.cuda = '11.8'
		mock_torch.version = mock_version

		mock_subprocess_run.side_effect = FileNotFoundError()

		# Initialize message first
		self.info.message = GPUInfo.nvidia_ready()

		get_nvidia_gpu_info('Linux', self.info)

		assert self.info.message.startswith(GPUInfo.nvidia_ready())
		assert GPUInfo.nvidia_smi_warning() in self.info.message

	@patch('app.features.hardware.api.subprocess.run')
	@patch('app.features.hardware.api.device_service')
	def test_nvidia_gpu_exists_but_cuda_unavailable(self, mock_device_service, mock_subprocess_run):
		"""Test NVIDIA GPU detected but CUDA unavailable (driver issue)."""
		mock_device_service.is_cuda = False

		# nvidia-smi succeeds (GPU exists)
		mock_result = MagicMock()
		mock_result.stdout = '525.60.11\n'
		mock_subprocess_run.return_value = mock_result

		get_nvidia_gpu_info('Linux', self.info)

		assert self.info.overall_status == GPUDriverStatusStates.DRIVER_ISSUE
		assert self.info.message == GPUInfo.nvidia_driver_issue()
		assert GPUInfo.nvidia_driver_status_step() in self.info.troubleshooting_steps


class TestGetMPSGPUInfo:
	"""Test Apple Silicon MPS GPU detection."""

	def setup_method(self):
		"""Set up test fixtures."""
		self.info = GPUDriverInfo(
			gpus=[],
			is_cuda=False,
			message='',
			overall_status=GPUDriverStatusStates.UNKNOWN_ERROR,
		)

	@patch('app.features.hardware.api.platform')
	def test_mps_gpu_ready_with_performance_warning(self, mock_platform):
		"""Test MPS GPU ready with performance warning."""
		mock_platform.machine.return_value = 'arm64'

		get_mps_gpu_info(self.info)

		assert self.info.overall_status == GPUDriverStatusStates.READY
		assert self.info.message == GPUInfo.macos_mps_ready()
		assert 'slower' in self.info.message or 'performance' in self.info.message.lower()
		assert self.info.macos_mps_available is True
		assert len(self.info.gpus) == 1
		assert self.info.gpus[0].name == 'arm64'
		assert self.info.gpus[0].is_primary is True


class TestGetSystemGPUInfo:
	"""Test main GPU detection logic for all platforms."""

	def setup_method(self):
		"""Clear cache before each test."""
		get_system_gpu_info.cache_clear()

	@patch('app.features.hardware.api.platform')
	@patch('app.features.hardware.api.device_service')
	@patch('app.features.hardware.api.subprocess')
	@patch('app.features.hardware.api.torch')
	def test_windows_with_cuda(self, mock_torch, mock_subprocess, mock_device_service, mock_platform):
		"""Test Windows with CUDA available."""
		mock_platform.system.return_value = 'Windows'
		mock_device_service.is_cuda = True
		mock_device_service.device_count = 1
		mock_device_service.get_device_name.return_value = 'NVIDIA GPU'
		mock_device_service.current_device = 0

		mock_device_property = MagicMock()
		mock_device_property.total_memory = 8 * 1024 * 1024 * 1024
		mock_device_property.major = 7
		mock_device_property.minor = 5
		mock_device_service.get_device_properties.return_value = mock_device_property

		mock_version = MagicMock()
		mock_version.cuda = '11.8'
		mock_torch.version = mock_version
		mock_torch.backends = MagicMock()

		mock_result = MagicMock()
		mock_result.stdout = '525.60.11\n'
		mock_subprocess.run.return_value = mock_result

		info = get_system_gpu_info()

		assert info.overall_status == GPUDriverStatusStates.READY
		assert info.message == GPUInfo.nvidia_ready()

	@patch('app.features.hardware.api.platform')
	@patch('app.features.hardware.api.device_service')
	@patch('app.features.hardware.api.subprocess')
	@patch('app.features.hardware.api.torch')
	def test_linux_with_cuda(self, mock_torch, mock_subprocess, mock_device_service, mock_platform):
		"""Test Linux with CUDA available."""
		mock_platform.system.return_value = 'Linux'
		mock_device_service.is_cuda = True
		mock_device_service.device_count = 1
		mock_device_service.get_device_name.return_value = 'NVIDIA GPU'
		mock_device_service.current_device = 0

		mock_device_property = MagicMock()
		mock_device_property.total_memory = 8 * 1024 * 1024 * 1024
		mock_device_property.major = 7
		mock_device_property.minor = 5
		mock_device_service.get_device_properties.return_value = mock_device_property

		mock_version = MagicMock()
		mock_version.cuda = '11.8'
		mock_torch.version = mock_version
		mock_torch.backends = MagicMock()

		mock_result = MagicMock()
		mock_result.stdout = '525.60.11\n'
		mock_subprocess.run.return_value = mock_result

		info = get_system_gpu_info()

		assert info.overall_status == GPUDriverStatusStates.READY
		assert info.message == GPUInfo.nvidia_ready()

	@patch('app.features.hardware.api.subprocess.run')
	@patch('app.features.hardware.api.torch')
	@patch('app.features.hardware.api.device_service')
	@patch('app.features.hardware.api.platform')
	def test_windows_without_cuda_shows_nvidia_troubleshooting(
		self, mock_platform, mock_device_service, mock_torch, mock_subprocess_run
	):
		"""Test Windows without CUDA shows NVIDIA troubleshooting."""
		mock_platform.system.return_value = 'Windows'
		mock_device_service.is_cuda = False
		mock_torch.backends = MagicMock()
		mock_subprocess_run.side_effect = FileNotFoundError()

		info = get_system_gpu_info()

		assert info.overall_status == GPUDriverStatusStates.NO_GPU
		assert info.message == GPUInfo.nvidia_no_gpu()
		assert info.recommendation_link == GPUInfo.nvidia_recommendation_link()
		assert info.troubleshooting_steps == GPUInfo.nvidia_troubleshooting_steps()

	@patch('app.features.hardware.api.subprocess.run')
	@patch('app.features.hardware.api.torch')
	@patch('app.features.hardware.api.device_service')
	@patch('app.features.hardware.api.platform')
	def test_linux_without_cuda_shows_nvidia_troubleshooting(
		self, mock_platform, mock_device_service, mock_torch, mock_subprocess_run
	):
		"""Test Linux without CUDA shows NVIDIA troubleshooting."""
		mock_platform.system.return_value = 'Linux'
		mock_device_service.is_cuda = False
		mock_torch.backends = MagicMock()
		mock_subprocess_run.side_effect = FileNotFoundError()

		info = get_system_gpu_info()

		assert info.overall_status == GPUDriverStatusStates.NO_GPU
		assert info.message == GPUInfo.nvidia_no_gpu()
		assert info.recommendation_link == GPUInfo.nvidia_recommendation_link()
		assert info.troubleshooting_steps == GPUInfo.nvidia_troubleshooting_steps()

	@patch('app.features.hardware.api.platform')
	@patch('app.features.hardware.api.device_service')
	@patch('app.features.hardware.api.torch')
	def test_macos_with_mps_shows_performance_warning(self, mock_torch, mock_device_service, mock_platform):
		"""Test macOS with MPS shows Apple Silicon message with performance warning."""
		mock_platform.system.return_value = 'Darwin'
		mock_platform.machine.return_value = 'arm64'
		mock_device_service.is_cuda = False

		mock_backends = MagicMock()
		mock_backends.mps.is_available.return_value = True
		mock_torch.backends = mock_backends

		info = get_system_gpu_info()

		assert info.overall_status == GPUDriverStatusStates.READY
		assert info.message == GPUInfo.macos_mps_ready()
		assert 'Apple Silicon' in info.message
		assert 'slower' in info.message or 'performance' in info.message.lower()
		assert info.macos_mps_available is True

	@patch('app.features.hardware.api.platform')
	@patch('app.features.hardware.api.device_service')
	@patch('app.features.hardware.api.torch')
	def test_macos_without_mps_intel_mac_no_acceleration(self, mock_torch, mock_device_service, mock_platform):
		"""Test macOS without MPS (Intel Mac) shows no acceleration message."""
		mock_platform.system.return_value = 'Darwin'
		mock_device_service.is_cuda = False

		mock_backends = MagicMock()
		mock_backends.mps.is_available.return_value = False
		mock_torch.backends = mock_backends

		info = get_system_gpu_info()

		assert info.overall_status == GPUDriverStatusStates.NO_GPU
		assert info.message == GPUInfo.macos_no_acceleration()
		assert info.macos_mps_available is False
		assert info.troubleshooting_steps == GPUInfo.macos_troubleshooting_steps()

	@patch('app.features.hardware.api.platform')
	@patch('app.features.hardware.api.device_service')
	@patch('app.features.hardware.api.torch')
	def test_macos_never_shows_nvidia_recommendations(self, mock_torch, mock_device_service, mock_platform):
		"""Test macOS NEVER shows NVIDIA recommendations regardless of state."""
		mock_platform.system.return_value = 'Darwin'
		mock_device_service.is_cuda = False

		# Test with MPS unavailable (Intel Mac)
		mock_backends = MagicMock()
		mock_backends.mps.is_available.return_value = False
		mock_torch.backends = mock_backends

		info = get_system_gpu_info()

		# Critical assertion: macOS should NEVER have NVIDIA recommendation link
		# Schema default is '' (empty string), so it should remain empty
		assert info.recommendation_link == '', 'macOS should not have NVIDIA recommendation link (should be empty)'

		# Troubleshooting steps should be macOS-specific, not NVIDIA
		if info.troubleshooting_steps:
			for step in info.troubleshooting_steps:
				# Verify no NVIDIA driver installation instructions
				assert 'NVIDIA' not in step or 'driver' not in step.lower(), (
					f'macOS should not show NVIDIA driver installation steps: {step}'
				)

	@patch('app.features.hardware.api.platform')
	@patch('app.features.hardware.api.device_service')
	@patch('app.features.hardware.api.torch')
	def test_unsupported_os(self, mock_torch, mock_device_service, mock_platform):
		"""Test unsupported operating system."""
		mock_platform.system.return_value = 'FreeBSD'
		mock_device_service.is_cuda = False
		mock_torch.backends = MagicMock()

		info = get_system_gpu_info()

		assert info.overall_status == GPUDriverStatusStates.NO_GPU
		assert info.message == GPUInfo.unsupported_os('FreeBSD')
		assert 'FreeBSD' in info.message

	@patch('app.features.hardware.api.platform')
	@patch('app.features.hardware.api.device_service')
	@patch('app.features.hardware.api.torch')
	def test_pytorch_import_error(self, mock_torch, mock_device_service, mock_platform):
		"""Test PyTorch ImportError."""
		mock_platform.system.return_value = 'Linux'
		mock_device_service.is_cuda = False

		# Make torch.backends raise ImportError when accessed
		type(mock_torch).backends = PropertyMock(side_effect=ImportError('PyTorch not found'))

		info = get_system_gpu_info()

		assert info.overall_status == GPUDriverStatusStates.UNKNOWN_ERROR
		assert info.message == GPUInfo.pytorch_not_installed()
		assert info.troubleshooting_steps == GPUInfo.pytorch_troubleshooting()

	@patch('app.features.hardware.api.platform')
	@patch('app.features.hardware.api.device_service')
	@patch('app.features.hardware.api.torch')
	@patch('app.features.hardware.api.logger')
	def test_unexpected_error(self, mock_logger, mock_torch, mock_device_service, mock_platform):
		"""Test unexpected exception during detection."""
		mock_platform.system.return_value = 'Linux'
		mock_device_service.is_cuda = False

		error_message = 'Unexpected runtime error'
		mock_torch.backends = MagicMock()
		with patch('app.features.hardware.api.get_nvidia_gpu_info', side_effect=RuntimeError(error_message)):
			info = get_system_gpu_info()

		assert info.overall_status == GPUDriverStatusStates.UNKNOWN_ERROR
		assert info.message == GPUInfo.unexpected_error(error_message)
		assert info.troubleshooting_steps == GPUInfo.error_troubleshooting_steps()
		mock_logger.error.assert_called_once()


class TestHardwareEndpoints:
	"""Test hardware API endpoints."""

	@patch('app.features.hardware.api.get_system_gpu_info')
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

	@patch('app.features.hardware.api.get_system_gpu_info')
	def test_recheck_endpoint_clears_cache(self, mock_get_gpu_info):
		"""Test GET /hardware/recheck clears cache and re-detects."""
		from app.features.hardware.api import recheck

		mock_info = GPUDriverInfo(
			gpus=[],
			is_cuda=True,
			message=GPUInfo.nvidia_ready(),
			overall_status=GPUDriverStatusStates.READY,
		)
		mock_get_gpu_info.return_value = mock_info

		# Mock cache_clear method
		with patch('app.features.hardware.api.get_system_gpu_info.cache_clear') as mock_cache_clear:
			result = recheck()

			mock_cache_clear.assert_called_once()
			assert result == mock_info

	@patch('app.features.hardware.api.add_device_index')
	def test_set_device_endpoint(self, mock_add_device_index):
		"""Test POST /hardware/device sets device index."""
		from app.features.hardware.api import set_device
		from app.schemas.hardware import SelectDeviceRequest

		mock_db = MagicMock()
		request = SelectDeviceRequest(device_index=1)

		result = set_device(request, db=mock_db)

		assert result['message'] == GPUInfo.device_set_success()
		assert result['device_index'] == 1
		mock_add_device_index.assert_called_once_with(mock_db, device_index=1)

	@patch('app.features.hardware.api.get_device_index')
	def test_get_device_endpoint(self, mock_get_device_index):
		"""Test GET /hardware/device returns current device index."""
		from app.features.hardware.api import get_device

		mock_db = MagicMock()
		mock_get_device_index.return_value = 0

		result = get_device(db=mock_db)

		assert result.device_index == 0
		mock_get_device_index.assert_called_once_with(mock_db)

	@patch('app.features.hardware.api.get_device_index')
	@patch('app.features.hardware.api.logger')
	def test_get_device_endpoint_error_handling(self, mock_logger, mock_get_device_index):
		"""Test GET /hardware/device handles errors."""
		from fastapi import HTTPException

		from app.features.hardware.api import get_device

		mock_db = MagicMock()
		error_msg = 'Database error'
		mock_get_device_index.side_effect = Exception(error_msg)

		with pytest.raises(HTTPException) as exc_info:
			get_device(db=mock_db)

		assert exc_info.value.status_code == 500
		assert error_msg in str(exc_info.value.detail)
		mock_logger.error.assert_called_once()

	@patch('app.features.hardware.api.add_max_memory')
	def test_set_max_memory_endpoint(self, mock_add_max_memory):
		"""Test POST /hardware/max-memory sets memory configuration."""
		from app.features.hardware.api import set_max_memory
		from app.schemas.hardware import MaxMemoryConfigRequest

		mock_db = MagicMock()
		config = MaxMemoryConfigRequest(ram_scale_factor=0.8, gpu_scale_factor=0.9)

		result = set_max_memory(config, db=mock_db)

		assert result['message'] == GPUInfo.memory_config_success()
		assert result['ram_scale_factor'] == 0.8
		assert result['gpu_scale_factor'] == 0.9
		mock_add_max_memory.assert_called_once_with(mock_db, ram_scale_factor=0.8, gpu_scale_factor=0.9)

	@patch('app.features.hardware.api.MemoryService')
	def test_get_device_memory_endpoint(self, mock_memory_service_class):
		"""Test GET /hardware/memory returns memory info."""
		from app.features.hardware.api import get_device_memory

		mock_db = MagicMock()
		mock_memory_service = MagicMock()
		mock_memory_service.total_gpu = 8192
		mock_memory_service.total_ram = 16384
		mock_memory_service_class.return_value = mock_memory_service

		result = get_device_memory(db=mock_db)

		assert result.gpu == 8192
		assert result.ram == 16384
		mock_memory_service_class.assert_called_once_with(mock_db)

	@patch('app.features.hardware.api.MemoryService')
	@patch('app.features.hardware.api.logger')
	def test_get_device_memory_endpoint_error_handling(self, mock_logger, mock_memory_service_class):
		"""Test GET /hardware/memory handles errors."""
		from fastapi import HTTPException

		from app.features.hardware.api import get_device_memory

		mock_db = MagicMock()
		error_msg = 'Memory service error'
		mock_memory_service_class.side_effect = Exception(error_msg)

		with pytest.raises(HTTPException) as exc_info:
			get_device_memory(db=mock_db)

		assert exc_info.value.status_code == 500
		assert error_msg in str(exc_info.value.detail)
		mock_logger.error.assert_called_once()
