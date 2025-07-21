"""Styles Blueprint"""

from fastapi import APIRouter

from app.predefined_styles import fooocus_styles, sai_styles
from app.services import styles_service

from .schemas import StyleSectionResponse

styles = APIRouter(
	prefix='/styles',
	tags=['styles'],
)


@styles.get('/')
def get_styles():
	"""List all styles"""

	return [
		StyleSectionResponse(
			id='fooocus',
			styles=fooocus_styles,
		),
		StyleSectionResponse(
			id='sai',
			styles=sai_styles,
		),
	]


@styles.get('/prompt')
def get_prompt_styles(user_prompt: str):
	return styles_service.apply_styles(user_prompt, ['fooocus_v2', 'fooocus_enhance', 'fooocus_sharp'])
