import logging
import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import get_db

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
    request: GenerateImageRequest, db: Session = Depends(get_db)
):
    """Generates an image based on the provided prompt and parameters. Returns the first generated image as a file."""
    try:
        filename = generator_service.generate_image(request, db)

        return FileResponse(
            filename, media_type='image/png', filename=os.path.basename(filename)
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@generators.get('/samplers')
def get_all_samplers():
    """
    Returns a list of available samplers for image generation.
    """

    return samplers
