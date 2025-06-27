"""Models Blueprint"""

import logging
from typing import Optional

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from fastapi import APIRouter, HTTPException, Query, status
from huggingface_hub import HfApi

from app.services.storage import get_model_dir

from .schemas import (
    LoadModelResponse,
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


@models.get('/load')
def load_model(id: str = Query(..., description='Model ID')):
    """Load model by id"""

    if not id:
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing 'id' query parameter",
        )

    try:
        model_dir = get_model_dir(id)
        pipeline = DiffusionPipeline.from_pretrained(model_dir)
        pipeline.to('cuda')

        return LoadModelResponse(
            id=id,
            config=dict(pipeline.config),
        )
    except Exception as e:
        logger.exception('Failed to load model')
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
