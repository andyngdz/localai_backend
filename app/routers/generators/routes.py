import logging

from fastapi import APIRouter

from app.routers.models.schemas import GenerateImageRequest
from app.services import model_manager

logger = logging.getLogger(__name__)
generators = APIRouter(
    prefix='/generators',
    tags=['generators'],
)


@generators.post('/')
async def generate_image(request: GenerateImageRequest):
    """
    Generates an image based on the provided prompt and parameters.
    Returns the generated image as a file.
    """
    pass


@generators.get('/available-samplers')
async def available_samplers():
    """
    Returns a list of available samplers for image generation.
    """

    samplers = model_manager.get_available_samplers()

    if not samplers:
        logger.warning('No available samplers found.')
        return []

    return samplers
