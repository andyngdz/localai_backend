"""Tests for image_processor module."""

import os
from datetime import datetime
from unittest.mock import patch

import pytest
import torch
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from PIL import Image


@pytest.fixture
def image_processor():
	"""Get the image_processor singleton."""
	from app.cores.generation.image_processor import image_processor

	return image_processor


class TestIsNsfwContentDetected:
	def test_returns_nsfw_flags_when_detected(self, image_processor):
		output = StableDiffusionPipelineOutput(nsfw_content_detected=[True, False, True], images=[None, None, None])

		result = image_processor.is_nsfw_content_detected(output)

		assert result == [True, False, True]

	def test_returns_false_list_when_no_nsfw(self, image_processor):
		output = StableDiffusionPipelineOutput(nsfw_content_detected=None, images=[None, None])

		result = image_processor.is_nsfw_content_detected(output)

		assert result == [False, False]

	def test_handles_empty_images_list(self, image_processor):
		output = StableDiffusionPipelineOutput(nsfw_content_detected=None, images=[])

		result = image_processor.is_nsfw_content_detected(output)

		assert result == []


class TestGenerateFileName:
	def test_generates_timestamp_based_filename(self, image_processor):
		filename = image_processor.generate_file_name()

		# Check format: YYYYMMDD_HHMMSS_ffffff (22 chars total)
		assert len(filename) == 22
		assert filename[8] == '_'
		assert filename[15] == '_'
		# Verify it's a valid datetime format
		datetime.strptime(filename, '%Y%m%d_%H%M%S_%f')


class TestSaveImage:
	def test_raises_value_error_when_image_is_none(self, image_processor):
		with pytest.raises(ValueError, match='Failed to generate any image'):
			image_processor.save_image(None)

	def test_saves_image_and_returns_paths(self, tmp_path, monkeypatch):
		# Create a test image
		test_image = Image.new('RGB', (64, 64), color='red')

		# Mock the paths
		generated_folder = tmp_path / 'generated'
		static_folder = tmp_path / 'static/generated_images'
		generated_folder.mkdir(parents=True, exist_ok=True)
		static_folder.mkdir(parents=True, exist_ok=True)

		# Monkeypatch at config level
		import sys

		img_proc_module = sys.modules['app.cores.generation.image_processor']
		monkeypatch.setattr(img_proc_module, 'GENERATED_IMAGES_FOLDER', str(generated_folder))
		monkeypatch.setattr(img_proc_module, 'GENERATED_IMAGES_STATIC_FOLDER', str(static_folder))

		# Now get the processor (after patching)
		from app.cores.generation.image_processor import image_processor

		static_path, file_name = image_processor.save_image(test_image)

		# Verify file was created
		assert os.path.exists(os.path.join(generated_folder, f'{file_name}.png'))
		assert static_path == os.path.join(str(static_folder), f'{file_name}.png')


class TestLatentsToRgb:
	def test_converts_latent_tensor_to_rgb_image(self, image_processor):
		# Create a real latent tensor (4 channels for latent space)
		latents = torch.randn(4, 8, 8)

		result = image_processor.latents_to_rgb(latents)

		assert isinstance(result, Image.Image)
		assert result.mode == 'RGB'
		assert result.size == (8, 8)  # Width x Height

	def test_handles_different_latent_sizes(self, image_processor):
		# Test with larger latent tensor
		latents = torch.randn(4, 16, 16)

		result = image_processor.latents_to_rgb(latents)

		assert isinstance(result, Image.Image)
		assert result.size == (16, 16)


class TestClearTensorCache:
	"""Test clear_tensor_cache() method."""

	def test_clears_cached_weights_and_biases(self, image_processor):
		"""Test clearing cached tensors (lines 58-70)."""
		# Setup - add cached tensors
		image_processor.cached_weights['layer1'] = torch.randn(10, 10)
		image_processor.cached_weights['layer2'] = torch.randn(20, 20)
		image_processor.cached_biases['bias1'] = torch.randn(10)

		# Execute
		image_processor.clear_tensor_cache()

		# Verify caches are empty
		assert len(image_processor.cached_weights) == 0
		assert len(image_processor.cached_biases) == 0

	@patch('app.cores.generation.image_processor.torch')
	def test_clears_cuda_cache_when_available(self, mock_torch, image_processor):
		"""Test CUDA cache clearing when CUDA is available (lines 69-70)."""
		# Setup
		mock_torch.cuda.is_available.return_value = True
		image_processor.cached_weights['test'] = torch.randn(5, 5)

		# Execute
		image_processor.clear_tensor_cache()

		# Verify CUDA cache was cleared
		mock_torch.cuda.empty_cache.assert_called_once()

	@patch('app.cores.generation.image_processor.torch')
	def test_skips_cuda_cache_when_not_available(self, mock_torch, image_processor):
		"""Test skips CUDA cache clearing when CUDA not available."""
		# Setup
		mock_torch.cuda.is_available.return_value = False
		image_processor.cached_weights['test'] = torch.randn(5, 5)

		# Execute
		image_processor.clear_tensor_cache()

		# Verify CUDA cache was NOT cleared
		mock_torch.cuda.empty_cache.assert_not_called()
