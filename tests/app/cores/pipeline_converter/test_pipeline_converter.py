from unittest.mock import MagicMock

import pytest

from app.cores.pipeline_converter.pipeline_converter import pipeline_converter


class TestPipelineConverter:
	"""Tests for PipelineConverter class."""

	def test_convert_to_img2img_with_none_pipe(self):
		"""Test that converting None pipe raises ValueError."""
		with pytest.raises(ValueError, match='Pipeline is None'):
			pipeline_converter.convert_to_img2img(None)

	def test_convert_to_text2img_with_none_pipe(self):
		"""Test that converting None pipe raises ValueError."""
		with pytest.raises(ValueError, match='Pipeline is None'):
			pipeline_converter.convert_to_text2img(None)

	def test_get_pipeline_type_with_none(self):
		"""Test pipeline type detection with None."""
		result = pipeline_converter.get_pipeline_type(None)
		assert result == 'unknown'

	def test_get_pipeline_type_text2img(self):
		"""Test pipeline type detection for text2img."""
		mock_pipe = MagicMock()
		# Remove image attribute to simulate text2img
		del mock_pipe.image

		result = pipeline_converter.get_pipeline_type(mock_pipe)
		assert result == 'text2img'

	def test_get_pipeline_type_img2img(self):
		"""Test pipeline type detection for img2img."""
		mock_pipe = MagicMock()
		# Has image attribute but not mask_image
		mock_pipe.image = MagicMock()
		del mock_pipe.mask_image

		result = pipeline_converter.get_pipeline_type(mock_pipe)
		assert result == 'img2img'

	def test_get_pipeline_type_inpainting(self):
		"""Test pipeline type detection for inpainting."""
		mock_pipe = MagicMock()
		# Has both image and mask_image attributes
		mock_pipe.image = MagicMock()
		mock_pipe.mask_image = MagicMock()

		result = pipeline_converter.get_pipeline_type(mock_pipe)
		assert result == 'inpainting'
