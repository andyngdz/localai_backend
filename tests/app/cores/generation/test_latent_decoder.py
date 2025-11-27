"""Tests for latent_decoder module."""

from typing import cast
from unittest.mock import Mock

import pytest
import torch
from PIL import Image


@pytest.fixture
def mock_pipe():
	"""Create a mock pipeline with all required components."""
	pipe = Mock()
	pipe.device = torch.device('cpu')
	pipe.dtype = torch.float32

	# Mock VAE
	pipe.vae = Mock()
	pipe.vae.config = Mock()
	pipe.vae.config.scaling_factor = 0.18215
	mock_decoder_output = Mock()
	mock_decoder_output.sample = Mock()
	pipe.vae.decode = Mock(return_value=mock_decoder_output)

	# Mock image processor
	pipe.image_processor = Mock()
	mock_image = Mock(spec=Image.Image)
	pipe.image_processor.postprocess = Mock(return_value=[mock_image])

	# Mock safety checker
	pipe.safety_checker = Mock(return_value=([mock_image], [False]))

	# Mock feature extractor
	pipe.feature_extractor = Mock()
	mock_features = Mock()
	mock_features.pixel_values = Mock()
	mock_features.pixel_values.to = Mock(return_value=mock_features.pixel_values)
	pipe.feature_extractor.return_value = mock_features
	pipe.feature_extractor.return_value.to = Mock(return_value=mock_features)

	return pipe


@pytest.fixture
def sample_latents():
	"""Create sample latent tensor."""
	return torch.randn(1, 4, 64, 64)


class TestDecodeLatents:
	"""Test decode_latents() method."""

	def test_decodes_latents_successfully(self, mock_pipe, sample_latents):
		"""Test successful latent decoding (happy path)."""
		from app.cores.generation.latent_decoder import latent_decoder

		result = latent_decoder.decode_latents(mock_pipe, sample_latents)

		# Verify VAE decode was called
		mock_pipe.vae.decode.assert_called_once()

		# Verify image processor postprocess was called
		mock_pipe.image_processor.postprocess.assert_called_once()

		# Verify result is list of images
		assert isinstance(result, list)
		assert len(result) > 0

	def test_scales_latents_before_decode(self, mock_pipe, sample_latents):
		"""Test that latents are scaled by VAE scaling factor."""
		from app.cores.generation.latent_decoder import latent_decoder

		latent_decoder.decode_latents(mock_pipe, sample_latents)

		# Get the tensor that was passed to vae.decode
		call_args = mock_pipe.vae.decode.call_args
		scaled_latents = call_args[0][0]

		# Verify scaling was applied
		expected_scale = 1.0 / mock_pipe.vae.config.scaling_factor
		expected_latents = sample_latents * expected_scale

		assert torch.allclose(scaled_latents, expected_latents, atol=1e-6)

	def test_calls_postprocess_with_pil_output_type(self, mock_pipe, sample_latents):
		"""Test that postprocess is called with output_type='pil'."""
		from app.cores.generation.latent_decoder import latent_decoder

		latent_decoder.decode_latents(mock_pipe, sample_latents)

		# Verify postprocess was called with correct output_type
		call_kwargs = mock_pipe.image_processor.postprocess.call_args[1]
		assert call_kwargs.get('output_type') == 'pil'


class TestRunSafetyChecker:
	"""Test run_safety_checker() method."""

	def test_runs_safety_checker_successfully(self, mock_pipe):
		"""Test successful safety checker execution (happy path)."""
		from app.cores.generation.latent_decoder import latent_decoder

		mock_image = Mock(spec=Image.Image)
		images = cast(list[Image.Image], [mock_image])

		result_images, nsfw_detected = latent_decoder.run_safety_checker(mock_pipe, images)

		# Verify safety checker was called
		mock_pipe.safety_checker.assert_called_once()

		# Verify feature extractor was called
		mock_pipe.feature_extractor.assert_called_once_with(images, return_tensors='pt')

		# Verify results
		assert isinstance(result_images, list)
		assert isinstance(nsfw_detected, list)
		assert len(nsfw_detected) == len(result_images)

	def test_returns_false_when_no_safety_checker(self, mock_pipe):
		"""Test handling when safety checker is None (SDXL case)."""
		from app.cores.generation.latent_decoder import latent_decoder

		mock_pipe.safety_checker = None
		mock_image = Mock(spec=Image.Image)
		images = cast(list[Image.Image], [mock_image, mock_image])

		result_images, nsfw_detected = latent_decoder.run_safety_checker(mock_pipe, images)

		# Verify images unchanged
		assert result_images == images

		# Verify all nsfw flags are False
		assert nsfw_detected == [False, False]

	def test_returns_false_when_no_feature_extractor(self, mock_pipe):
		"""Test handling when feature extractor is None (edge case)."""
		from app.cores.generation.latent_decoder import latent_decoder

		mock_pipe.feature_extractor = None
		mock_image = Mock(spec=Image.Image)
		images = cast(list[Image.Image], [mock_image])

		result_images, nsfw_detected = latent_decoder.run_safety_checker(mock_pipe, images)

		# Verify images unchanged
		assert result_images == images

		# Verify nsfw flag is False
		assert nsfw_detected == [False]

	def test_logs_warning_when_nsfw_detected(self, mock_pipe, caplog):
		"""Test that warning is logged when NSFW content detected."""
		from app.cores.generation.latent_decoder import latent_decoder

		# Configure safety checker to detect NSFW in first image
		mock_image = Mock(spec=Image.Image)
		images = cast(list[Image.Image], [mock_image, mock_image])
		mock_pipe.safety_checker.return_value = (images, [True, False])

		with caplog.at_level('WARNING'):
			latent_decoder.run_safety_checker(mock_pipe, images)

		# Verify warning was logged
		assert 'NSFW content detected' in caplog.text
		assert '1 of 2' in caplog.text

	def test_logs_info_when_no_nsfw_detected(self, mock_pipe, caplog):
		"""Test that info is logged when no NSFW content detected."""
		from app.cores.generation.latent_decoder import latent_decoder

		mock_image = Mock(spec=Image.Image)
		images = cast(list[Image.Image], [mock_image])
		mock_pipe.safety_checker.return_value = (images, [False])

		with caplog.at_level('INFO'):
			latent_decoder.run_safety_checker(mock_pipe, images)

		# Verify info was logged
		assert 'No NSFW content detected' in caplog.text

	def test_converts_tensors_to_correct_device_and_dtype(self, mock_pipe):
		"""Test that feature extractor output is moved to correct device and dtype."""
		from app.cores.generation.latent_decoder import latent_decoder

		mock_image = Mock(spec=Image.Image)
		images = cast(list[Image.Image], [mock_image])

		latent_decoder.run_safety_checker(mock_pipe, images)

		# Verify feature extractor output was moved to pipe device
		mock_pipe.feature_extractor.return_value.to.assert_called_once_with(mock_pipe.device)

		# Verify pixel values were converted to pipe dtype
		call_kwargs = mock_pipe.safety_checker.call_args[1]
		assert 'clip_input' in call_kwargs
