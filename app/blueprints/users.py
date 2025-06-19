"""Users Blueprint"""

import requests
from flask import Blueprint, Response, abort

users = Blueprint("users", __name__)


@users.route("/avatar/<string:user_id>.png", methods=["GET"])
def get_user_avatar(user_id):
    """Proxy and serve the Hugging Face user avatar"""
    try:
        response = requests.get(
            f"https://huggingface.co/api/users/{user_id}/avatar", timeout=5
        )
        response.raise_for_status()
        avatar_url = response.json().get("avatarUrl")

        if not avatar_url:
            return abort(404, description="Avatar not found.")

        avatar_image_response = requests.get(avatar_url, timeout=5)
        avatar_image_response.raise_for_status()

        return Response(
            avatar_image_response.content,
            content_type=avatar_image_response.headers.get("Content-Type", "image/png"),
        )

    except requests.RequestException as e:
        return abort(500, description=f"Error fetching avatar: {e}")
