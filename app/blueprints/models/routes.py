"""Models Blueprint"""

import logging

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from flask import Blueprint, jsonify, request
from huggingface_hub import HfApi

from app.schemas.core import ErrorResponse, ErrorType
from app.services.storage import get_model_dir

from .schemas import (
    LoadModelResponse,
    ModelSearchInfo,
    ModelSearchInfoListResponse,
)

logger = logging.getLogger(__name__)
models = Blueprint('models', __name__)
api = HfApi()

default_limit = 50
default_pipeline_tag = 'text-to-image'
default_sort = 'downloads'


@models.route('/search', methods=['GET'])
def list_models():
    """List models from Hugging Face Hub."""

    model_name_param = request.args.get('model_name')
    filter_param = request.args.get('filter')
    limit_param = request.args.get('limit', type=int, default=default_limit)

    hf_models_generator = api.list_models(
        full=True,
        filter=filter_param,
        limit=limit_param,
        model_name=model_name_param,
        pipeline_tag=default_pipeline_tag,
        sort=default_sort,
    )

    models = list(hf_models_generator)

    models_search_info = []

    for model in models:
        model_search_info = ModelSearchInfo(**model.__dict__)
        models_search_info.append(model_search_info)

    return jsonify(
        ModelSearchInfoListResponse(models_search_info=models_search_info).model_dump()
    ), 200


@models.route('/', methods=['GET'])
def get_model_info():
    """Get model info by model's id"""
    id = request.args.get('id')

    if not id:
        return jsonify(
            ErrorResponse(
                detail="Missing 'id' query parameter", type=ErrorType.ValidationError
            )
        ), 400

    model_info = api.model_info(id, files_metadata=True)

    return jsonify(model_info), 200


@models.route('/load', methods=['GET'])
def load_model():
    """Load model by id"""
    id = request.args.get('id')

    if not id:
        return jsonify(
            ErrorResponse(
                detail="Missing 'id' query parameter", type=ErrorType.ValidationError
            )
        ), 400

    try:
        model_dir = get_model_dir(id)
        pipeline = DiffusionPipeline.from_pretrained(model_dir)
        pipeline.to('cuda')

        return jsonify(
            LoadModelResponse(
                id=id,
                config=dict(pipeline.config),
            ).model_dump()
        ), 200
    except Exception as e:
        logger.exception('Failed to load model')
        return jsonify(
            ErrorResponse(detail=str(e), type=ErrorType.InternalServerError)
        ), 500
