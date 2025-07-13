import base64
import io

from PIL import Image


class ImageService:
    def to_base64(self, image: Image.Image) -> str:
        """
        Convert a PIL Image to a base64 encoded string.

        Args:
            image (Image): The PIL Image to convert.

        Returns:
            str: Base64 encoded string of the image.
        """

        buffered = io.BytesIO()
        image.save(buffered, format='png')
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return image_base64


image_service = ImageService()
