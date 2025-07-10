"""Styles Blueprint"""

from fastapi import APIRouter

from app.predefined_styles import fooocus_prompts, sai_prompts

from .schemas import StylesResponse

styles = APIRouter(
    prefix='/styles',
    tags=['styles'],
)


@styles.get('/')
def get_styles():
    """List all styles"""

    return StylesResponse(
        fooocus=fooocus_prompts,
        sai=sai_prompts,
    )
