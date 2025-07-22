import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import database_service
from app.database.crud import add_history, get_histories
from app.model_manager import model_manager_service
from app.schemas.generators import GeneratorConfig

logger = logging.getLogger(__name__)
histories = APIRouter(
	prefix='/histories',
	tags=['histories'],
)


@histories.post('/')
async def new_history(
	config: GeneratorConfig,
	db: Session = Depends(database_service.get_db),
):
	"""Add a new history entry for the image generation request."""
	try:
		if model_manager_service.id is None:
			raise HTTPException(
				status_code=status.HTTP_400_BAD_REQUEST,
				detail='No model loaded. Please load a model before creating a history entry.',
			)

		history = add_history(db, model_manager_service.id, config)

		return history.id

	except ValueError as error:
		return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error))


@histories.get('/')
async def all_histories(db: Session = Depends(database_service.get_db)):
	"""Get all histories."""
	try:
		histories = get_histories(db)

		return histories
	except Exception as e:
		logger.error(f'Error fetching histories: {e}')
		raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Could not fetch histories')
