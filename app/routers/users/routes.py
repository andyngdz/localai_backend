"""Users Router"""

import requests
from fastapi import APIRouter
from fastapi.responses import FileResponse, Response

users = APIRouter(
	prefix='/users',
	tags=['users'],
)

PLACEHOLDER_IMAGE_PATH = 'static/empty.png'


@users.get('/avatar/{user_id}.png')
def get_user_avatar(user_id: str):
	"""Proxy and serve the Hugging Face user avatar"""
	try:
		response = requests.get(f'https://huggingface.co/api/users/{user_id}/avatar', timeout=5)
		if response.status_code == 404:
			response = requests.get(f'https://huggingface.co/api/organizations/{user_id}/avatar', timeout=5)

		response.raise_for_status()

		avatar_url = response.json().get('avatarUrl')

		if not avatar_url or avatar_url.endswith('.svg'):
			return FileResponse(PLACEHOLDER_IMAGE_PATH)

		avatar_image_response = requests.get(avatar_url, timeout=5)
		avatar_image_response.raise_for_status()

		return Response(content=avatar_image_response.content, media_type='image/png')

	except requests.RequestException:
		return FileResponse(PLACEHOLDER_IMAGE_PATH)
