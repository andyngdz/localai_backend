import os

from config import BASE_MODEL_DIR


def get_model_dir(id: str) -> str:
    """
    Get the directory path for a model based on its ID.

    Args:
        id (str): The ID of the model.

    Returns:
        str: The directory path where the model is stored.
    """
    return os.path.join(BASE_MODEL_DIR, id.replace('/', '--'))
