"""Tests for progress_callback module."""

from unittest.mock import Mock, patch

import pytest
import torch
from PIL import Image


@pytest.fixture
def mock_progress_callback():
	"""Create ProgressCallback with mocked dependencies."""
	with (
		patch('app.cores.generation.progress_callback.image_processor') as mock_image_processor,
		patch('app.cores.generation.progress_callback.image_service') as mock_image_service,
		patch('app.cores.generation.progress_callback.socket_service') as mock_socket_service,
	):
		# Configure mocks
		test_image = Image.new('RGB', (8, 8), color='blue')
		mock_image_processor.latents_to_rgb.return_value = test_image
		mock_image_service.to_base64.return_value = 'base64_encoded_image'

		from app.cores.generation.progress_callback import ProgressCallback

		callback = ProgressCallback()

		yield callback, mock_image_processor, mock_image_service, mock_socket_service


class TestCallbackOnStepEnd:
	def test_processes_latents_and_emits_socket_events(self, mock_progress_callback):
		callback, mock_image_processor, mock_image_service, mock_socket_service = mock_progress_callback

		# Create mock pipe and latents
		mock_pipe = Mock()
		latents = [torch.randn(4, 8, 8), torch.randn(4, 8, 8)]  # 2 images in batch
		callback_kwargs = {'latents': latents}

		result = callback.callback_on_step_end(mock_pipe, current_step=5, timestep=0.5, callback_kwargs=callback_kwargs)

		# Verify latents_to_rgb was called for each latent
		assert mock_image_processor.latents_to_rgb.call_count == 2

		# Verify image_service.to_base64 was called for each image
		assert mock_image_service.to_base64.call_count == 2

		# Verify socket emissions
		assert mock_socket_service.image_generation_step_end.call_count == 2

		# Verify callback_kwargs is returned unchanged
		assert result == callback_kwargs

	def test_handles_single_image_batch(self, mock_progress_callback):
		callback, mock_image_processor, mock_image_service, mock_socket_service = mock_progress_callback

		mock_pipe = Mock()
		latents = [torch.randn(4, 8, 8)]  # Single image
		callback_kwargs = {'latents': latents}

		result = callback.callback_on_step_end(mock_pipe, current_step=1, timestep=0.9, callback_kwargs=callback_kwargs)

		# Verify single image processing
		assert mock_image_processor.latents_to_rgb.call_count == 1
		assert mock_socket_service.image_generation_step_end.call_count == 1
		assert result == callback_kwargs
