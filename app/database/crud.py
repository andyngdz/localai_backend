import os
from logging import getLogger
from typing import List

from sqlalchemy.orm import Session, selectinload

from app.database.models import Config, GeneratedImage, History, Model
from app.schemas.generators import GeneratorConfig, ImageGenerationResponse

from .constant import DEFAULT_MAX_GPU_SCALE_FACTOR, DEFAULT_MAX_RAM_SCALE_FACTOR, DeviceSelection

logger = getLogger(__name__)


def add_model(db: Session, model_id: str, model_dir: str):
	"""Add a model to the database or update its local path if it already exists."""
	model = db.query(Model).filter(Model.model_id == model_id).first()

	if model:
		model.model_dir = model_dir
	else:
		model = Model(model_id=model_id, model_dir=model_dir)
		db.add(model)

	db.commit()

	return model


def downloaded_models(db: Session) -> List[Model]:
	"""Get all models from the database."""
	models = db.query(Model).all()

	return models


def is_model_downloaded(db: Session, model_id: str) -> bool:
	"""Check if a model is downloaded (status 'completed') in the database."""
	model = db.query(Model).filter(Model.model_id == model_id).first()

	return model is not None


def get_device_index(db: Session) -> int:
	"""Get selected device index from the database."""
	config = db.query(Config).first()

	return config.device_index if config else DeviceSelection.NOT_FOUND


def add_device_index(db: Session, device_index: int):
	"""Add or update selected device"""

	config = db.query(Config).first()

	if config:
		config.device_index = device_index
	else:
		config = Config(device_index=device_index)
		db.add(config)

	db.commit()

	return config


def add_max_memory(db: Session, ram_scale_factor: float, gpu_scale_factor: float):
	"""Add or update configuration in the database."""
	config = db.query(Config).first()

	if config:
		config.ram_scale_factor = ram_scale_factor
		config.gpu_scale_factor = gpu_scale_factor
	else:
		config = Config(ram_scale_factor=ram_scale_factor, gpu_scale_factor=gpu_scale_factor)
		db.add(config)

	db.commit()

	return config


def get_gpu_scale_factor(db: Session) -> float:
	"""Get GPU max factor from the database."""

	config = db.query(Config).first()

	if config and config.gpu_scale_factor is not None:
		return config.gpu_scale_factor

	return DEFAULT_MAX_GPU_SCALE_FACTOR


def get_ram_scale_factor(db: Session) -> float:
	"""Get RAM max factor from the database."""

	config = db.query(Config).first()

	if config and config.ram_scale_factor is not None:
		return config.gpu_scale_factor

	return DEFAULT_MAX_RAM_SCALE_FACTOR


def add_history(db: Session, model: str, config: GeneratorConfig):
	"""Add a history entry to the database."""

	history = History(
		prompt=config.prompt,
		model=model,
		config=config.model_dump(),
	)
	db.add(history)
	db.commit()

	return history


def get_histories(db: Session):
	"""Get all histories and their associated generated images from the database."""
	histories = db.query(History).options(selectinload(History.generated_images)).all()

	return histories


def delete_history_entry(db: Session, history_id: int):
	"""Delete a history entry and its associated generated images."""

	with db.begin():
		try:
			history = db.query(History).filter(History.id == history_id).first()

			if not history:
				raise ValueError(f'History entry with id {history_id} does not exist.')

			delete_images = db.query(GeneratedImage).filter(GeneratedImage.history_id == history.id).all()

			db.delete(history)
			db.commit()

			logger.info(f'Deleted history entry with id: {history_id}')

			for image in delete_images:
				path = image.path

				if os.path.exists(path):
					logger.info(f'Deleting image file: {path}')
					os.remove(path)

		except Exception as error:
			raise ValueError(f'Error deleting history entry: {str(error)}')

	return history_id


def add_generated_image(db: Session, history_id: int, response: ImageGenerationResponse):
	"""Add a generated image to the history entry."""
	items = response.items
	nsfw_content_detected = response.nsfw_content_detected

	for index, item in enumerate(items):
		is_nsfw = nsfw_content_detected[index]

		generated_image = GeneratedImage(
			history_id=history_id,
			path=item.path,
			file_name=item.file_name,
			is_nsfw=is_nsfw,
		)

		db.add(generated_image)

	db.commit()
