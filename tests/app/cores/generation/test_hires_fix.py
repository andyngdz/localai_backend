"""Tests for hires fix processor."""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from PIL import Image

from app.cores.generation.hires_fix import HiresFixProcessor
from app.schemas.generators import GeneratorConfig, OutputType
from app.schemas.hires_fix import HiresFixConfig, UpscalerType


class TestHiresFixProcessor:
	"""Test hires fix processor."""

	@pytest.fixture
	def processor(self):
		"""Create processor instance."""
		return HiresFixProcessor()

	@pytest.fixture
	def mock_pipe(self):
		"""Create mock pipeline that returns PIL images."""
		pipe = MagicMock()
		pipe.device = torch.device('cpu')
		output = MagicMock()
		output.images = [Image.new('RGB', (1024, 1024), color='red')]
		pipe.return_value = output
		return pipe

	@pytest.fixture
	def sample_latents(self):
		"""Create sample latent tensor."""
		return torch.randn(1, 4, 64, 64)

	@pytest.fixture
	def generator_config(self):
		"""Create generator config."""
		return GeneratorConfig(
			prompt='test prompt',
			negative_prompt='test negative',
			width=512,
			height=512,
			steps=20,
			cfg_scale=7.5,
			clip_skip=1,
			seed=42,
			hires_fix=HiresFixConfig(
				upscale_factor=2.0,
				upscaler=UpscalerType.LANCZOS,
				denoising_strength=0.7,
				steps=15,
			),
		)

	@pytest.fixture
	def torch_generator(self):
		"""Create torch generator."""
		return torch.Generator().manual_seed(42)

	def test_get_steps_uses_hires_steps_when_nonzero(self, processor):
		"""Test that _get_steps uses hires_steps when > 0."""
		result = processor._get_steps(hires_steps=15, base_steps=20)
		assert result == 15

	def test_get_steps_uses_base_steps_when_zero(self, processor):
		"""Test that _get_steps uses base_steps when hires_steps is 0."""
		result = processor._get_steps(hires_steps=0, base_steps=20)
		assert result == 20

	def test_apply_requires_hires_fix_config(self, processor, mock_pipe, sample_latents, torch_generator):
		"""Test that apply asserts if hires_fix is None."""
		config = GeneratorConfig(
			prompt='test',
			width=512,
			height=512,
			steps=20,
			hires_fix=None,
		)

		with pytest.raises(AssertionError):
			processor.apply(config, mock_pipe, sample_latents, torch_generator)

	def test_apply_decodes_latents_and_upscales(
		self, processor, mock_pipe, sample_latents, generator_config, torch_generator
	):
		"""Test that apply decodes latents and upscales in pixel space."""
		with (
			patch('app.cores.generation.hires_fix.latent_decoder') as mock_decoder,
			patch('app.cores.generation.hires_fix.image_upscaler') as mock_upscaler,
		):
			base_images = [Image.new('RGB', (512, 512), color='blue')]
			upscaled_images = [Image.new('RGB', (1024, 1024), color='green')]

			mock_decoder.decode_latents.return_value = base_images
			mock_upscaler.upscale.return_value = upscaled_images

			processor.apply(generator_config, mock_pipe, sample_latents, torch_generator)

			mock_decoder.decode_latents.assert_called_once()
			mock_upscaler.upscale.assert_called_once_with(
				base_images,
				scale_factor=2.0,
				upscaler_type=UpscalerType.LANCZOS,
			)

	def test_apply_calls_pipeline_with_correct_params(
		self, processor, mock_pipe, sample_latents, generator_config, torch_generator
	):
		"""Test that apply calls pipeline with correct parameters."""
		with (
			patch('app.cores.generation.hires_fix.latent_decoder') as mock_decoder,
			patch('app.cores.generation.hires_fix.image_upscaler') as mock_upscaler,
		):
			base_images = [Image.new('RGB', (512, 512), color='blue')]
			upscaled_images = [Image.new('RGB', (1024, 1024), color='green')]

			mock_decoder.decode_latents.return_value = base_images
			mock_upscaler.upscale.return_value = upscaled_images

			processor.apply(generator_config, mock_pipe, sample_latents, torch_generator)

			mock_pipe.assert_called_once()
			call_kwargs = mock_pipe.call_args[1]

			assert call_kwargs['prompt'] == 'test prompt'
			assert call_kwargs['negative_prompt'] == 'test negative'
			assert call_kwargs['num_inference_steps'] == 15
			assert call_kwargs['strength'] == 0.7
			assert call_kwargs['guidance_scale'] == 7.5
			assert call_kwargs['clip_skip'] == 1
			assert call_kwargs['output_type'] == OutputType.PIL.value
			assert call_kwargs['image'] == upscaled_images

	def test_apply_uses_base_steps_when_hires_steps_zero(self, processor, mock_pipe, sample_latents, torch_generator):
		"""Test that apply uses base steps when hires_steps is 0."""
		with (
			patch('app.cores.generation.hires_fix.latent_decoder') as mock_decoder,
			patch('app.cores.generation.hires_fix.image_upscaler') as mock_upscaler,
		):
			mock_decoder.decode_latents.return_value = [Image.new('RGB', (512, 512))]
			mock_upscaler.upscale.return_value = [Image.new('RGB', (1024, 1024))]

			config = GeneratorConfig(
				prompt='test',
				width=512,
				height=512,
				steps=20,
				hires_fix=HiresFixConfig(
					upscale_factor=2.0,
					upscaler=UpscalerType.LANCZOS,
					denoising_strength=0.7,
					steps=0,
				),
			)

			processor.apply(config, mock_pipe, sample_latents, torch_generator)

			call_kwargs = mock_pipe.call_args[1]
			assert call_kwargs['num_inference_steps'] == 20

	def test_apply_returns_refined_images(self, processor, mock_pipe, sample_latents, generator_config, torch_generator):
		"""Test that apply returns refined PIL images from pipeline."""
		with (
			patch('app.cores.generation.hires_fix.latent_decoder') as mock_decoder,
			patch('app.cores.generation.hires_fix.image_upscaler') as mock_upscaler,
		):
			mock_decoder.decode_latents.return_value = [Image.new('RGB', (512, 512))]
			mock_upscaler.upscale.return_value = [Image.new('RGB', (1024, 1024))]

			expected_images = [Image.new('RGB', (1024, 1024), color='red')]
			mock_output = Mock()
			mock_output.images = expected_images
			mock_pipe.return_value = mock_output

			result = processor.apply(generator_config, mock_pipe, sample_latents, torch_generator)

			assert result == expected_images
			assert all(isinstance(img, Image.Image) for img in result)
