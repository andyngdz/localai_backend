import logging
from flask import Blueprint, jsonify
from huggingface_hub import HfApi
from app.schemas.model_info import ModelInfo, ModelInfoListResponse

models = Blueprint("models", __name__)
api = HfApi()


@models.route("/list", methods=["GET"])
def list_models():
    hf_models_generator = api.list_models(filter="stable-diffusion", limit=50)
    hf_models = list(hf_models_generator)
    models_info = []
    for m in hf_models:
        model_info = ModelInfo(
            id=m.id,
            name=m.id,
            author=m.author,
            downloads=m.downloads,
        )
        models_info.append(model_info)

    return jsonify(ModelInfoListResponse(models_info=models_info).model_dump())
