"""Constants for img2img feature.

DEPRECATED: Import from app.constants.img2img instead.
This module re-exports for backwards compatibility.
"""

from app.constants.img2img import IMG2IMG_DEFAULT_STRENGTH, IMG2IMG_RESIZE_MODES

# Legacy constant - use app.services.styles.DEFAULT_NEGATIVE_PROMPT instead
DEFAULT_NEGATIVE_PROMPT = 'blurry, bad quality, low resolution, distorted'

__all__ = ['IMG2IMG_DEFAULT_STRENGTH', 'IMG2IMG_RESIZE_MODES', 'DEFAULT_NEGATIVE_PROMPT']
