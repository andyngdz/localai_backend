"""Styles Blueprint"""

from fastapi import APIRouter

from app.predefined_styles import fooocus_styles, sai_styles

from .schemas import StylesSectionResponse

styles = APIRouter(
    prefix='/styles',
    tags=['styles'],
)


@styles.get('/')
def get_styles():
    """List all styles"""

    return [
        StylesSectionResponse(
            id='fooocus',
            styles=fooocus_styles,
        ),
        StylesSectionResponse(
            id='sai',
            styles=sai_styles,
        ),
    ]
