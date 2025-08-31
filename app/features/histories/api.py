"""Image Generation History Router"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.cores.model_manager import model_manager
from app.database import database_service
from app.database.crud import add_history, delete_history_entry, get_histories
from app.schemas.generators import GeneratorConfig
from app.schemas.responses import JSONResponseMessage

logger = logging.getLogger(__name__)
histories = APIRouter(
	prefix='/histories',
	tags=['histories'],
)


@histories.post('/')
async def add_new_history(
	config: GeneratorConfig,
	db: Session = Depends(database_service.get_db),
):
	"""Add a new history entry for the image generation request."""
	try:
		if model_manager.id is None:
			raise HTTPException(
				status_code=status.HTTP_400_BAD_REQUEST,
				detail='No model loaded. Please load a model before creating a history entry.',
			)

		history = add_history(db, model_manager.id, config)

		return history.id

	except ValueError as error:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error))


@histories.get('/')
async def all_histories(db: Session = Depends(database_service.get_db)):
	"""Get all histories."""
	try:
		histories = get_histories(db)

		return histories
	except ValueError as error:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error))
	except Exception:
		raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Could not fetch histories')


@histories.delete('/{history_id}')
async def delete_history(history_id: int, db: Session = Depends(database_service.get_db)):
	"""Delete a history entry."""
	try:
		delete_history_entry(db, history_id)

		return JSONResponseMessage(message=f'History entry deleted successfully {history_id}')
	except Exception as error:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error))
