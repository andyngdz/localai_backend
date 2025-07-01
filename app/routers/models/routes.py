"""Models Blueprint"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from huggingface_hub import HfApi
from sqlalchemy.orm import Session

from app.database import get_db
from app.database.crud import check_if_model_downloaded, get_all_downloaded_models
from app.services import get_model_dir, model_manager

from .schemas import (
    LoadModelRequest,
    LoadModelResponse,
    ModelAvailableResponse,
    ModelDownloadedResponse,
    ModelSearchInfo,
    ModelSearchInfoListResponse,
)

logger = logging.getLogger(__name__)
models = APIRouter(
    prefix='/models',
    tags=['models'],
)
api = HfApi()

default_limit = 50
default_pipeline_tag = 'text-to-image'
default_sort = 'downloads'


@models.get('/search', response_model=ModelSearchInfoListResponse)
def list_models(
    model_name: Optional[str] = Query(None),
    filter: Optional[str] = Query(None),
    limit: int = Query(10),
):
    """List models from Hugging Face Hub."""

    hf_models_generator = api.list_models(
        full=True,
        filter=filter,
        limit=limit,
        model_name=model_name,
        pipeline_tag=default_pipeline_tag,
        sort=default_sort,
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
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing 'id' query parameter",
        )

    model_info = api.model_info(id, files_metadata=True)

    return model_info


@models.get('/downloaded')
def get_downloaded_models(db: Session = Depends(get_db)):
    """Get a list of downloaded models"""
    try:
        models = get_all_downloaded_models(db)

        return ModelDownloadedResponse(models=models).model_dump()
    except Exception as e:
        logger.exception('Failed to fetch available models')
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@models.get('/available')
def check_if_model_already_downloaded(
    id: str = Query(..., description='Model ID'), db: Session = Depends(get_db)
):
    """Check if model is already downloaded by id"""

    try:
        is_downloaded = check_if_model_downloaded(db, id)
        return ModelAvailableResponse(id=id, is_downloaded=is_downloaded)
    except Exception as e:
        logger.exception('Failed to check if model is downloaded')
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@models.post('/load', response_model=LoadModelResponse)
def load_model(request: LoadModelRequest):
    """Load model by id"""
    try:
        model_id = request.id
        model_dir = get_model_dir(model_id)
        model_config = model_manager.load_model(model_id, model_dir)
        return LoadModelResponse(
            id=model_id,
            config=model_config,
        )

    except FileNotFoundError as e:
        logger.error(f'Model file not found for {model_id}: {e}')
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model files not found for ID '{model_id}'. {e}",
        )
    except Exception as e:
        logger.exception(f'Failed to load model {model_id}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model '{model_id}': {e}",
        )


@models.post('/unload')
async def unload_model():
    """
    Unloads the current model from memory.
    """

    try:
        model_manager.unload_model()
        return {'message': 'Model unloaded successfully.'}
    except Exception as e:
        logger.exception('Failed to unload model')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to unload model: {e}',
        )


@models.get('/generate')
def generate_image(prompt: str):
    pass
