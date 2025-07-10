"""Styles Blueprint"""

from fastapi import APIRouter

from app.predefined_styles import fooocus_prompt, sai_prompt

styles = APIRouter(
    prefix='/styles',
    tags=['styles'],
)


@styles.get('/')
def get_styles():
    """List all styles"""

    return [fooocus_prompt, sai_prompt]
