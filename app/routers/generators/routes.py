import logging
import os
import uuid

import torch
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from app.services import model_manager

from .schemas import GenerateImageRequest

logger = logging.getLogger(__name__)
generators = APIRouter(
    prefix='/generators',
    tags=['generators'],
)


@generators.post('/start')
async def start_generation_image(request: GenerateImageRequest):
    """
    Generates an image based on the provided prompt and parameters.
    Returns the first generated image as a file.

    Note: 'hires_fix' is acknowledged but not fully implemented in this MVP.
    Batch generation (batch_count > 1) is currently not fully utilized for the response,
    only 'batch_size' images are generated per prompt, and only the first is returned.
    """

    logger.info(f'Received image generation request: {request}')

    pipe = model_manager.pipe

    if pipe is None:
        logger.warning('Attempted to generate image, but no model is loaded.')
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='No model is currently loaded. Please load a model first using /models/load.',
        )

    try:
        # 1. Set the sampler dynamically for this generation
        # This will update the pipe's scheduler to the requested one

        # model_manager.set_sampler(request.sampler)
        # logger.info(f'Sampler set to: {request.sampler.value}')

        # 2. Set random seed for reproducibility if provided
        random_seed = None  # Ensure random_seed is always defined

        if request.seed != -1:
            torch.manual_seed(request.seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed(request.seed)

            logger.info(f'Using random seed: {request.seed}')

            random_seed = request.seed
        else:
            # If seed is -1, generate a random seed for this run
            random_seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
            torch.manual_seed(random_seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_seed)

            logger.info(f'Using auto-generated random seed: {random_seed}')

        logger.info(
            f"Generating image(s) for prompt: '{request.prompt}' "
            f'with steps={request.steps}, CFG={request.cfg_scale}, '
            f'size={request.width}x{request.height}, '
            f'batch_size={request.batch_size}, batch_count={request.batch_count}'
        )

        # 3. Perform image generation
        # The `pipe` call typically takes `num_images_per_prompt`
        # which aligns with your `batch_size`.
        # For `batch_count`, you would typically loop this generation
        # or handle it at a higher level if you need multiple distinct batches.

        # Hires fix is a more advanced feature (e.g., img2img on upscaled latent).
        # For simplicity in this MVP, we'll generate directly at requested height/width.
        # If request.hires_fix is True, you would implement a multi-step process here.
        if request.hires_fix:
            logger.warning(
                'Hires fix requested, but not fully implemented in this MVP. Generating directly at requested resolution.'
            )

        generation_output = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.steps,
            guidance_scale=request.cfg_scale,
            height=request.height,
            width=request.width,
            num_images_per_prompt=request.batch_size,
            generator=torch.Generator(device=pipe.device).manual_seed(
                request.seed if request.seed != -1 else random_seed
            ),
        )  # type: ignore

        # Handle potential safety checker output
        # Some pipelines return 'nsfw_content_detected'
        if (
            hasattr(generation_output, 'nsfw_content_detected')
            and generation_output.nsfw_content_detected is not None
            and any(generation_output.nsfw_content_detected)
        ):
            logger.warning('NSFW content detected. Returning a blank image or error.')
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Generated image flagged as NSFW by safety checker.',
            )

        # For FileResponse, we can only return one image.
        # We'll return the first image from the generated batch.
        image = generation_output.images[0]

        # 4. Save the image to a temporary file
        temp_dir = './.localai_generated_images'
        os.makedirs(temp_dir, exist_ok=True)  # Ensure directory exists

        filename = os.path.join(temp_dir, f'{uuid.uuid4().hex}.png')
        image.save(filename)
        logger.info(f'Generated image saved to: {filename}')

        # 5. Return the image file as a FileResponse
        return FileResponse(
            filename, media_type='image/png', filename=os.path.basename(filename)
        )

    except FileNotFoundError as e:
        logger.error(f'Model directory not found: {e}')
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f'Model files not found: {e}'
        )
    except Exception as e:
        logger.exception(f'Failed to generate image for prompt: "{request.prompt}"')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to generate image: {e}',
        )


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
