"""Styles Blueprint"""

from fastapi import APIRouter

from app.predefined_styles import fooocus_prompts, sai_prompts

styles = APIRouter(
    prefix='/styles',
    tags=['styles'],
)


@styles.get('/')
def get_styles():
    """List all styles"""

    return [fooocus_prompts, sai_prompts]
