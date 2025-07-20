"""Models Blueprint"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from huggingface_hub import HfApi
from sqlalchemy.orm import Session

from app.database import database_service
from app.database.crud import downloaded_models, is_model_downloaded
from app.model_manager import model_manager_service

from .schemas import (
	LoadModelRequest,
	LoadModelResponse,
	ModelAvailableResponse,
	ModelSearchInfo,
	ModelSearchInfoListResponse,
)

logger = logging.getLogger(__name__)
models = APIRouter(
	prefix='/models',
	tags=['models'],
)
api = HfApi()


@models.get('/search')
def list_models(
	model_name: Optional[str] = Query(
		default=None,
		description='Model name to search for',
	),
	filter: Optional[str] = Query(default=None, description='Filter for models'),
	limit: int = Query(default=20, description='Number of models to return'),
	sort: Optional[str] = Query(default='likes', description='Sort order for models'),
):
	"""List models from Hugging Face Hub."""

	hf_models_generator = api.list_models(
		full=True,
		filter=filter,
		limit=limit,
		model_name=model_name,
		pipeline_tag='text-to-image',
		sort=sort,
	)

	models = list(hf_models_generator)
	models_search_info = []

	for model in models:
		model_search_info = ModelSearchInfo(**model.__dict__)
		models_search_info.append(model_search_info)

	return ModelSearchInfoListResponse(models_search_info=models_search_info)


@models.get('/details')
def get_model_info(id: str = Query(..., description='Model ID')):
	"""Get model info by model's id"""
	if not id:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail="Missing 'id' query parameter",
		)

	model_info = api.model_info(id, files_metadata=True)

	return model_info


@models.get('/downloaded')
def get_downloaded_models(db: Session = Depends(database_service.get_db)):
	"""Get a list of downloaded models"""
	try:
		models = downloaded_models(db)

		return models
	except Exception as error:
		logger.exception('Failed to fetch available models')

		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail=str(error),
		)


@models.get('/available')
def is_model_already_downloaded(
	id: str = Query(..., description='Model ID'),
	db: Session = Depends(database_service.get_db),
):
	"""Check if model is already downloaded by id"""

	try:
		is_downloaded = is_model_downloaded(db, id)

		return ModelAvailableResponse(id=id, is_downloaded=is_downloaded)
	except Exception as error:
		logger.exception('Failed to check if model is downloaded')

		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail=str(error),
		)


@models.post('/load')
async def load_model(request: LoadModelRequest):
	"""Load model by id"""
	id = None

	try:
		id = request.id

		config = await model_manager_service.load_model_async(id)
		sample_size = model_manager_service.get_sample_size()

		return LoadModelResponse(id=id, config=config, sample_size=sample_size)

	except FileNotFoundError as error:
		logger.error(f'Model file not found for {id}: {error}')

		raise HTTPException(
			status_code=status.HTTP_404_NOT_FOUND,
			detail=f"Model files not found for ID '{id}'. {error}",
		)
	except Exception as error:
		logger.exception(f'Failed to load model {id}')

		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail=f"Failed to load model '{id}': {error}",
		)


@models.post('/unload')
def unload_model():
	"""
	Unloads the current model from memory.
	"""

	try:
		model_manager_service.unload_model()
	except Exception as error:
		logger.exception('Failed to unload model')

		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail=f'Failed to unload model: {error}',
		)
