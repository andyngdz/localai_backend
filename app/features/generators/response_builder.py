"""Response building for image generation results."""

from typing import Any

from app.cores.generation.image_utils import process_generated_images
from app.schemas.generators import ImageGenerationResponse


class ResponseBuilder:
	"""Builds ImageGenerationResponse from pipeline output."""

	def build_response(self, output: Any) -> ImageGenerationResponse:
		"""Build response from pipeline output.

		Args:
			output: Pipeline output containing generated images

		Returns:
			ImageGenerationResponse with processed images and NSFW detection
		"""
		items, nsfw_content_detected = process_generated_images(output)

		return ImageGenerationResponse(
			items=items,
			nsfw_content_detected=nsfw_content_detected,
		)


response_builder = ResponseBuilder()
