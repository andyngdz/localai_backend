from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import config_crud
from app.database.service import database_service
from app.features.config.service import config_service
from app.schemas.config import ConfigResponse, SafetyCheckRequest

config = APIRouter(prefix='/config', tags=['config'])


@config.get('/', response_model=ConfigResponse)
def get_config(db: Session = Depends(database_service.get_db)) -> ConfigResponse:
	"""Returns application configuration for the frontend."""
	return config_service.get_config(db)


@config.put('/safety-check', response_model=ConfigResponse)
def update_safety_check(request: SafetyCheckRequest, db: Session = Depends(database_service.get_db)) -> ConfigResponse:
	"""Update safety check setting."""
	config_crud.set_safety_check_enabled(db, request.enabled)

	return config_service.get_config(db)
