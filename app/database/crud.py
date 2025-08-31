import os
from logging import getLogger
from typing import List

from sqlalchemy.orm import Session, selectinload

from app.database.models import GeneratedImage, History, Model
from app.schemas.generators import GeneratorConfig, ImageGenerationResponse

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
