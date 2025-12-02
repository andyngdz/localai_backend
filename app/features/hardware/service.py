"""Hardware service for GPU detection and device configuration."""

from sqlalchemy.orm import Session

from app.database.config_crud import add_device_index, add_max_memory, get_device_index
from app.schemas.hardware import (
	GetCurrentDeviceIndex,
	GPUDriverInfo,
	MaxMemoryConfigRequest,
	MemoryResponse,
)
from app.services import logger_service
from app.services.memory import MemoryService

from .gpu_detector import GPUDetector
from .info import GPUInfo

logger = logger_service.get_logger(__name__, category='Hardware')


class HardwareService:
	"""Orchestrates hardware detection and configuration.

	This service manages:
	- GPU detection and information
	- Device selection
	- Memory configuration
	"""

	def __init__(self):
		self.gpu_detector = GPUDetector()

	def get_gpu_info(self) -> GPUDriverInfo:
		"""Get cached GPU information.

		Returns:
			GPUDriverInfo with detected hardware information
		"""
		return self.gpu_detector.detect()

	def recheck_gpu_info(self) -> GPUDriverInfo:
		"""Force re-detection of GPU.

		Returns:
			GPUDriverInfo with freshly detected hardware information
		"""
		self.gpu_detector.clear_cache()
		logger.info('Forcing re-check of GPU driver status by clearing cache.')
		return self.gpu_detector.detect()

	def get_memory_info(self, db: Session) -> MemoryResponse:
		"""Get memory configuration.

		Args:
			db: Database session

		Returns:
			MemoryResponse with GPU and RAM memory information
		"""
		memory_service = MemoryService(db)

		return MemoryResponse(
			gpu=memory_service.total_gpu,
			ram=memory_service.total_ram,
		)

	def set_device(self, db: Session, device_index: int) -> dict:
		"""Set active device.

		Args:
			db: Database session
			device_index: Index of the device to select

		Returns:
			Success message with device_index
		"""
		add_device_index(db, device_index=device_index)

		return {'message': GPUInfo.device_set_success(), 'device_index': device_index}

	def get_device(self, db: Session) -> GetCurrentDeviceIndex:
		"""Get current device.

		Args:
			db: Database session

		Returns:
			GetCurrentDeviceIndex with current device index
		"""
		device_index = get_device_index(db)

		return GetCurrentDeviceIndex(device_index=device_index)

	def set_max_memory(self, db: Session, config: MaxMemoryConfigRequest) -> dict:
		"""Set max memory configuration.

		Args:
			db: Database session
			config: Memory configuration request

		Returns:
			Success message with configuration values
		"""
		add_max_memory(db, ram_scale_factor=config.ram_scale_factor, gpu_scale_factor=config.gpu_scale_factor)

		return {
			'message': GPUInfo.memory_config_success(),
			'ram_scale_factor': config.ram_scale_factor,
			'gpu_scale_factor': config.gpu_scale_factor,
		}


hardware_service = HardwareService()
