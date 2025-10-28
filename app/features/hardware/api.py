"""Hardware Router"""

import functools
import platform
import subprocess
import sys

import torch
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import database_service
from app.database.config_crud import add_device_index, add_max_memory, get_device_index
from app.services import device_service, logger_service
from app.services.memory import MemoryService

from .info import GPUInfo
from .schemas import (
	GetCurrentDeviceIndex,
	GPUDeviceInfo,
	GPUDriverInfo,
	GPUDriverStatusStates,
	MaxMemoryConfigRequest,
	MemoryResponse,
	SelectDeviceRequest,
)

logger = logger_service.get_logger(__name__, category='API')

hardware = APIRouter(
	prefix='/hardware',
	tags=['hardware'],
)

if sys.platform == 'win32':
	CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW
else:
	CREATE_NO_WINDOW = 0


def default_gpu_info() -> GPUDriverInfo:
	"""Initializes GPUDriverInfo with default unknown status."""
	return GPUDriverInfo(
		gpus=[],
		is_cuda=device_service.is_cuda,
		message=GPUInfo.default_detecting(),
		overall_status=GPUDriverStatusStates.UNKNOWN_ERROR,
	)


def get_nvidia_gpu_info(system_os: str, info: GPUDriverInfo):
	if device_service.is_cuda:
		info.overall_status = GPUDriverStatusStates.READY
		info.message = GPUInfo.nvidia_ready()
		version = getattr(torch, 'version', None)
		info.cuda_runtime_version = getattr(version, 'cuda', None)

		gpus = []
		device_count = device_service.device_count

		for index in range(device_count):
			name = device_service.get_device_name(index)
			device_property = device_service.get_device_properties(index)

			if device_property is None:
				logger.warning(f'Could not retrieve properties for device index {index}. Skipping.')
				continue

			total_memory = device_property.total_memory
			cuda_compute_capability = f'{device_property.major}.{device_property.minor}'
			is_primary = index == device_service.current_device

			gpus.append(
				GPUDeviceInfo(
					name=name,
					memory=total_memory,
					cuda_compute_capability=cuda_compute_capability,
					is_primary=is_primary,
				)
			)

		info.gpus = gpus

		try:
			result = subprocess.run(
				[
					'nvidia-smi',
					'--query-gpu=driver_version',
					'--format=csv,noheader',
				],
				capture_output=True,
				text=True,
				check=True,
				creationflags=(CREATE_NO_WINDOW if system_os == 'Windows' else 0),
			)
			driver_version = result.stdout.strip().split('\n')[0]
			info.nvidia_driver_version = driver_version
		except (subprocess.CalledProcessError, FileNotFoundError) as error:
			logger.warning(f'nvidia-smi not found or failed: {error}. Cannot get detailed driver version.')
			info.message += GPUInfo.nvidia_smi_warning()
	else:
		info.overall_status = GPUDriverStatusStates.NO_GPU

		# Only show NVIDIA troubleshooting steps for non-macOS systems
		# macOS does not support NVIDIA CUDA
		if system_os != 'Darwin':
			info.message = GPUInfo.nvidia_no_gpu()
			info.recommendation_link = GPUInfo.nvidia_recommendation_link()
			info.troubleshooting_steps = GPUInfo.nvidia_troubleshooting_steps()
			try:
				subprocess.run(
					['nvidia-smi'],
					capture_output=True,
					text=True,
					check=True,
					creationflags=(CREATE_NO_WINDOW if system_os == 'Windows' else 0),
				)
				info.overall_status = GPUDriverStatusStates.DRIVER_ISSUE
				info.message = GPUInfo.nvidia_driver_issue()
				info.troubleshooting_steps.insert(0, GPUInfo.nvidia_driver_status_step())
			except (subprocess.CalledProcessError, FileNotFoundError):
				pass
		else:
			# macOS doesn't support NVIDIA CUDA
			info.message = GPUInfo.macos_no_acceleration()


def get_mps_gpu_info(info: GPUDriverInfo):
	info.overall_status = GPUDriverStatusStates.READY
	info.message = GPUInfo.macos_mps_ready()
	info.macos_mps_available = True
	info.gpus.append(GPUDeviceInfo(name=platform.machine(), is_primary=True))


@functools.cache
def get_system_gpu_info() -> GPUDriverInfo:
	"""
	Detects GPU and driver information based on the operating system.
	This function will be called by the endpoints.
	"""

	info = default_gpu_info()

	try:
		system_os = platform.system()
		backends = torch.backends

		if system_os in ('Windows', 'Linux'):
			get_nvidia_gpu_info(system_os, info)
		elif system_os == 'Darwin':
			if hasattr(backends, 'mps') and backends.mps.is_available():
				get_mps_gpu_info(info)
			else:
				info.overall_status = GPUDriverStatusStates.NO_GPU
				info.message = GPUInfo.macos_no_acceleration()
				info.macos_mps_available = False
				info.troubleshooting_steps = GPUInfo.macos_troubleshooting_steps()
		else:
			info.overall_status = GPUDriverStatusStates.NO_GPU
			info.message = GPUInfo.unsupported_os(system_os)

	except ImportError:
		info.overall_status = GPUDriverStatusStates.UNKNOWN_ERROR
		info.message = GPUInfo.pytorch_not_installed()
		info.troubleshooting_steps = GPUInfo.pytorch_troubleshooting()
	except (subprocess.SubprocessError, OSError, AttributeError, RuntimeError) as error:
		logger.error(f'Error during GPU detection: {error}')
		info.overall_status = GPUDriverStatusStates.UNKNOWN_ERROR
		info.message = GPUInfo.unexpected_error(str(error))
		info.troubleshooting_steps = GPUInfo.error_troubleshooting_steps()

	return info


@hardware.get('/')
def get_hardware():
	"""
	Returns the current GPU and driver status of the system.
	This will return the cached result of _get_system_gpu_info().
	"""
	driver_info = get_system_gpu_info()

	return driver_info


@hardware.get('/memory')
def get_device_memory(db: Session = Depends(database_service.get_db)):
	"""
	Returns the maximum memory configuration for the system.
	This will return the cached result of _get_system_gpu_info().
	"""
	try:
		memory_service = MemoryService(db)

		return MemoryResponse(
			gpu=memory_service.total_gpu,
			ram=memory_service.total_ram,
		)
	except Exception as error:
		logger.error(f'Error retrieving maximum memory configuration: {error}')

		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail=str(error),
		)


@hardware.get('/recheck')
def recheck():
	"""
	Forces the backend to re-evaluate and update the GPU and driver status.
	This clears the cache of _get_system_gpu_info() and then calls it again.
	"""
	get_system_gpu_info.cache_clear()  # Clear the cache

	logger.info('Forcing re-check of GPU driver status by clearing cache.')
	driver_info = get_system_gpu_info()  # Re-run detection (will now actually execute)

	return driver_info


@hardware.post('/device')
def set_device(request: SelectDeviceRequest, db: Session = Depends(database_service.get_db)):
	"""Select device"""

	device_index = request.device_index
	add_device_index(db, device_index=device_index)

	return {'message': GPUInfo.device_set_success(), 'device_index': device_index}


@hardware.get('/device')
def get_device(db: Session = Depends(database_service.get_db)):
	"""Get current selected device"""
	try:
		device_index = get_device_index(db)

		return GetCurrentDeviceIndex(device_index=device_index)
	except Exception as error:
		logger.error(f'Error retrieving current selected device: {error}')

		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail=str(error),
		)


@hardware.post('/max-memory')
def set_max_memory(config: MaxMemoryConfigRequest, db: Session = Depends(database_service.get_db)):
	"""Set maximum memory configuration."""

	add_max_memory(db, ram_scale_factor=config.ram_scale_factor, gpu_scale_factor=config.gpu_scale_factor)

	return {
		'message': GPUInfo.memory_config_success(),
		'ram_scale_factor': config.ram_scale_factor,
		'gpu_scale_factor': config.gpu_scale_factor,
	}
