"""Models Blueprint"""

from flask import Blueprint, jsonify, request
from huggingface_hub import HfApi

from app.schemas.models import ModelSearchInfo, ModelSearchInfoListResponse

models = Blueprint("models", __name__)
api = HfApi()


@models.route("/search", methods=["GET"])
def list_models():
    """List models from Hugging Face Hub."""

    filter_param = request.args.get("filter", default="stable-diffusion")
    limit_param = request.args.get("limit", type=int, default=50)

    hf_models_generator_params = {
        "pipeline_tag": "text-to-image",
        "filter": filter_param,
        "limit": limit_param,
    }

    hf_models_generator = api.list_models(**hf_models_generator_params)
    hf_models = list(hf_models_generator)

    models_search_info = []

    for m in hf_models:
        model_search_info = ModelSearchInfo(
            id=m.id,
            name=m.id,
            author=m.author,
            downloads=m.downloads,
            likes=m.likes,
            trending_score=m.trending_score,
            tags=m.tags,
        )
        models_search_info.append(model_search_info)

    return jsonify(
        ModelSearchInfoListResponse(models_search_info=models_search_info).model_dump()
    ), 200
