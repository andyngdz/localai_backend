import os

from config import BASE_CACHE_DIR


def get_model_dir(id: str) -> str:
    """
    Get the directory path for a model based on its ID.

    Args:
        id (str): The ID of the model.

    Returns:
        str: The directory path where the model is stored.
    """
    name_serialized = id.replace('/', '--')
    dir = os.path.join(BASE_CACHE_DIR, f'models--{name_serialized}')

    return dir
