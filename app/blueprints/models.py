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

    hf_models_generator = api.list_models(
        pipeline_tag="text-to-image",
        sort="donwloads",
        filter=filter_param,
        limit=limit_param,
        full=True,
        fetch_config=True,
        cardData=True,
    )
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


@models.route("/<string:id>", methods=["GET"])
def get_model_info(id):
    """Get model info by model's id"""
    model_info = api.model_info(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", files_metadata=True
    )

    return jsonify(model_info), 200
