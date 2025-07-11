import os

from config import BASE_CACHE_DIR, BASE_CACHE_LOCK_DIR


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


def get_model_lock_dir(id: str) -> str:
    """
    Get the full path for a model file based on its ID and filename.

    Args:
        id (str): The ID of the model.
        filename (str): The name of the file within the model directory.

    Returns:
        str: The full path to the model file.
    """
    name_serialized = id.replace('/', '--')

    dir = os.path.join(BASE_CACHE_LOCK_DIR, f'models--{name_serialized}')

    return dir
