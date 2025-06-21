"""Models Blueprint"""

from flask import Blueprint, jsonify, request
from huggingface_hub import HfApi, ModelCard

from app.schemas.core import ErrorResponse, ErrorType
from app.schemas.models import (
    ModelSearchInfo,
    ModelSearchInfoListResponse,
)

models = Blueprint("models", __name__)
api = HfApi()

default_filter = "stable-diffusion"
default_limit = 50
default_pipeline_tag = "text-to-image"
default_sort = "downloads"


@models.route("/search", methods=["GET"])
def list_models():
    """List models from Hugging Face Hub."""

    model_name_param = request.args.get("model_name")
    filter_param = request.args.get("filter", default=default_filter)
    limit_param = request.args.get("limit", type=int, default=default_limit)

    print(filter_param)

    hf_models_generator = api.list_models(
        filter=filter_param,
        full=True,
        limit=limit_param,
        model_name=model_name_param,
        pipeline_tag=default_pipeline_tag,
        sort=default_sort,
    )

    hf_models = list(hf_models_generator)
    models_search_info = []

    for m in hf_models:
        model_search_info = ModelSearchInfo(**m.__dict__)
        models_search_info.append(model_search_info)

    return jsonify(
        ModelSearchInfoListResponse(models_search_info=models_search_info).model_dump()
    ), 200


@models.route("/", methods=["GET"])
def get_model_info():
    """Get model info by model's id"""
    id = request.args.get("id")

    if not id:
        return jsonify(
            ErrorResponse(
                detail="Missing 'id' query parameter", type=ErrorType.ValidationError
            )
        ), 400

    model_card = ModelCard.load(id)
    model_info = api.model_info(id, files_metadata=True)

    return jsonify({"model_info": model_info, "content": model_card.text}), 200
