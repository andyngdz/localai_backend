"""Users Blueprint"""

import requests
from flask import Blueprint, Response, send_file

users = Blueprint('users', __name__)
PLACEHOLDER_IMAGE_PATH = 'static/empty.png'


@users.route('/avatar/<string:user_id>.png', methods=['GET'])
def get_user_avatar(user_id):
    """Proxy and serve the Hugging Face user avatar"""
    try:
        response = requests.get(
            f'https://huggingface.co/api/users/{user_id}/avatar', timeout=5
        )
        if response.status_code == 404:
            response = requests.get(
                f'https://huggingface.co/api/organizations/{user_id}/avatar', timeout=5
            )

        response.raise_for_status()

        avatar_url = response.json().get('avatarUrl')

        if not avatar_url or avatar_url.endswith('.svg'):
            return send_file(PLACEHOLDER_IMAGE_PATH)

        avatar_image_response = requests.get(avatar_url, timeout=5)
        avatar_image_response.raise_for_status()

        return Response(avatar_image_response.content)

    except requests.RequestException:
        return send_file(PLACEHOLDER_IMAGE_PATH)
