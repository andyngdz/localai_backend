import logging
import os

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse

from .constants import samplers
from .schemas import GenerateImageRequest
from .services import generator_service

logger = logging.getLogger(__name__)
generators = APIRouter(
    prefix='/generators',
    tags=['generators'],
)


@generators.post('/')
async def start_generation_image(
    request: GenerateImageRequest,
    id: str = Query(..., description='Socket ID for tracking the generation process'),
):
    """Generates an image based on the provided prompt and parameters. Returns the first generated image as a file."""
    try:
        filename = generator_service.generate_image(id, request)

        return FileResponse(
            filename, media_type='image/png', filename=os.path.basename(filename)
        )
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error))


@generators.get('/samplers')
def get_all_samplers():
    """
    Returns a list of available samplers for image generation.
    """

    return samplers
