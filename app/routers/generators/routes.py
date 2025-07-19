import logging
import os

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.constants import constant_service
from app.database import database_service
from app.database.crud import add_history
from app.model_manager import model_manager_service

from app.schemas.generators import ImageGenerationRequest
from .service import generator_service

logger = logging.getLogger(__name__)
generators = APIRouter(
    prefix='/generators',
    tags=['generators'],
)


@generators.post('/')
async def start_generation_image(
    request: ImageGenerationRequest,
    db: Session = Depends(database_service.get_db),
):
    """Generates an image based on the provided prompt and parameters. Returns the first generated image as a file."""
    try:
        if model_manager_service.id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='No model loaded. Please load a model before generating images.',
            )

        filename = await generator_service.generate_image(request)

        add_history(db, model_manager_service.id, request)

        return FileResponse(
            filename,
            media_type='image/png',
            filename=os.path.basename(filename),
        )

    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error))


@generators.get('/samplers')
async def get_all_samplers():
    """
    Returns a list of available samplers for image generation.
    """

    return constant_service.samplers
