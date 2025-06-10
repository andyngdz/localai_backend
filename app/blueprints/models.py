"""Models Blueprint"""

from flask import Blueprint, jsonify, request
from huggingface_hub import HfApi

from app.schemas.model_info import ModelInfo, ModelInfoListResponse

models = Blueprint("models", __name__)
api = HfApi()


@models.route("/search", methods=["GET"])
def list_models():
    """List models from Hugging Face Hub."""

    filter_param = request.args.get("filter", default="stable-diffusion")
    limit_param = request.args.get("limit", type=int, default=50)

    hf_models_generator_params = {
        "filter": filter_param,
        "limit": limit_param,
    }

    hf_models_generator = api.list_models(**hf_models_generator_params)
    hf_models = list(hf_models_generator)
    models_info = []
    for m in hf_models:
        model_info = ModelInfo(
            id=m.id,
            name=m.id,
            author=m.author,
            downloads=m.downloads,
            likes=m.likes,
            trending_score=m.trending_score,
            tags=m.tags,
        )
        models_info.append(model_info)

    return jsonify(ModelInfoListResponse(models_info=models_info).model_dump()), 200
