from enum import StrEnum

import pydash

from app.schemas.model_loader import ModelLoaderProgressStep

SAFETY_CHECKER_MODEL = 'CompVis/stable-diffusion-safety-checker'
CLIP_IMAGE_PROCESSOR_MODEL = 'openai/clip-vit-base-patch32'


_MODEL_LOADING_PROGRESS_MESSAGES = (
	(1, 'Initializing model loader...'),
	(2, 'Loading feature extractor...'),
	(3, 'Checking model cache...'),
	(4, 'Preparing loading strategies...'),
)


MODEL_LOADING_PROGRESS_STEPS: list[ModelLoaderProgressStep] = pydash.map_(
	_MODEL_LOADING_PROGRESS_MESSAGES,
	lambda entry: ModelLoaderProgressStep(id=entry[0], message=entry[1]),
)


class ModelLoadingStrategy(StrEnum):
	"""Enum for model loading strategy types."""

	SINGLE_FILE = 'single_file'
	PRETRAINED = 'pretrained'
