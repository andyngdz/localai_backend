"""Styles Blueprint"""

from fastapi import APIRouter
from predefined_styles.fooocus import fooocus_prompt
from predefined_styles.sai import sai_prompt

styles = APIRouter(
    prefix='/styles',
    tags=['styles'],
)


@styles.get('/')
def get_styles():
    """List all styles"""

    return [fooocus_prompt, sai_prompt]
