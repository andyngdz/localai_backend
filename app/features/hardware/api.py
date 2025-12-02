"""Hardware Router"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import database_service
from app.schemas.hardware import MaxMemoryConfigRequest, SelectDeviceRequest
from app.services import logger_service

from .service import hardware_service

logger = logger_service.get_logger(__name__, category='API')

hardware = APIRouter(
	prefix='/hardware',
	tags=['hardware'],
)


@hardware.get('/')
def get_hardware():
	"""
	Returns the current GPU and driver status of the system.
	This will return the cached result of GPU detection.
	"""
	driver_info = hardware_service.get_gpu_info()

	return driver_info


@hardware.get('/memory')
def get_device_memory(db: Session = Depends(database_service.get_db)):
	"""
	Returns the maximum memory configuration for the system.
	"""
	try:
		return hardware_service.get_memory_info(db)
	except Exception as error:
		logger.error(f'Error retrieving maximum memory configuration: {error}')

		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail=str(error),
		) from error


@hardware.get('/recheck')
def recheck():
	"""
	Forces the backend to re-evaluate and update the GPU and driver status.
	This clears the cache and re-runs GPU detection.
	"""
	driver_info = hardware_service.recheck_gpu_info()

	return driver_info


@hardware.post('/device')
def set_device(request: SelectDeviceRequest, db: Session = Depends(database_service.get_db)):
	"""Select device"""
	device_index = request.device_index

	return hardware_service.set_device(db, device_index)


@hardware.get('/device')
def get_device(db: Session = Depends(database_service.get_db)):
	"""Get current selected device"""
	try:
		return hardware_service.get_device(db)
	except Exception as error:
		logger.error(f'Error retrieving current selected device: {error}')

		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail=str(error),
		) from error


@hardware.post('/max-memory')
def set_max_memory(config: MaxMemoryConfigRequest, db: Session = Depends(database_service.get_db)):
	"""Set maximum memory configuration."""
	return hardware_service.set_max_memory(db, config)
