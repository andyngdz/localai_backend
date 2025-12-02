from fastapi import APIRouter

from app.features.config.service import config_service
from app.schemas.config import ConfigResponse

config = APIRouter(prefix='/config', tags=['config'])


@config.get('/', response_model=ConfigResponse)
def get_config() -> ConfigResponse:
	"""Returns application configuration for the frontend."""
	return ConfigResponse(
		upscalers=config_service.get_upscaler_sections(),
	)
