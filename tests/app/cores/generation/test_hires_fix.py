"""Tests for hires fix processor."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from app.cores.generation.hires_fix import HiresFixProcessor
from app.schemas.generators import GeneratorConfig
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
	def sample_images(self):
		"""Create sample PIL images."""
		return [Image.new('RGB', (512, 512), color='blue')]

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

	def test_apply_requires_hires_fix_config(self, processor, mock_pipe, sample_images, torch_generator):
		"""Test that apply asserts if hires_fix is None."""
		config = GeneratorConfig(
			prompt='test',
			width=512,
			height=512,
			steps=20,
			hires_fix=None,
		)

		with pytest.raises(AssertionError):
			processor.apply(config, mock_pipe, torch_generator, sample_images)

	def test_traditional_upscaler_delegates_to_upscale(
		self, processor, mock_pipe, sample_images, generator_config, torch_generator
	):
		"""Test that traditional upscalers delegate to upscale."""
		with patch('app.cores.generation.hires_fix.traditional_upscaler') as mock_upscaler:
			refined_images = [Image.new('RGB', (1024, 1024), color='green')]
			mock_upscaler.upscale.return_value = refined_images

			result = processor.apply(generator_config, mock_pipe, torch_generator, sample_images)

			mock_upscaler.upscale.assert_called_once_with(
				generator_config,
				mock_pipe,
				torch_generator,
				sample_images,
				scale_factor=2.0,
				upscaler_type=UpscalerType.LANCZOS,
				hires_steps=15,
				denoising_strength=0.7,
			)
			assert result == refined_images

	def test_apply_passes_hires_steps_zero_to_upscaler(self, processor, mock_pipe, sample_images, torch_generator):
		"""Test that apply passes hires_steps=0 to upscaler (upscaler resolves to base steps)."""
		with patch('app.cores.generation.hires_fix.traditional_upscaler') as mock_upscaler:
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

			processor.apply(config, mock_pipe, torch_generator, sample_images)

			call_kwargs = mock_upscaler.upscale.call_args[1]
			assert call_kwargs['hires_steps'] == 0

	def test_ai_upscaler_delegates_to_realesrgan(self, processor, mock_pipe, sample_images, torch_generator):
		"""Test that AI upscalers delegate to realesrgan_upscaler."""
		with patch('app.cores.generation.hires_fix.realesrgan_upscaler') as mock_realesrgan:
			upscaled_images = [Image.new('RGB', (1024, 1024), color='green')]
			mock_realesrgan.upscale.return_value = upscaled_images

			config = GeneratorConfig(
				prompt='test',
				width=512,
				height=512,
				steps=20,
				hires_fix=HiresFixConfig(
					upscale_factor=2.0,
					upscaler=UpscalerType.REALESRGAN_X4PLUS,
					denoising_strength=0.7,
					steps=15,
				),
			)

			result = processor.apply(config, mock_pipe, torch_generator, sample_images)

			mock_realesrgan.upscale.assert_called_once_with(
				sample_images,
				UpscalerType.REALESRGAN_X4PLUS,
				2.0,
			)
			mock_pipe.assert_not_called()
			assert result == upscaled_images

	def test_all_realesrgan_variants_skip_refinement(self, processor, mock_pipe, sample_images, torch_generator):
		"""Test that all Real-ESRGAN variants skip img2img refinement."""
		realesrgan_upscalers = [
			UpscalerType.REALESRGAN_X2PLUS,
			UpscalerType.REALESRGAN_X4PLUS,
			UpscalerType.REALESRGAN_X4PLUS_ANIME,
		]

		for upscaler_type in realesrgan_upscalers:
			mock_pipe.reset_mock()

			with patch('app.cores.generation.hires_fix.realesrgan_upscaler') as mock_realesrgan:
				upscaled_images = [Image.new('RGB', (1024, 1024), color='green')]
				mock_realesrgan.upscale.return_value = upscaled_images

				config = GeneratorConfig(
					prompt='test',
					width=512,
					height=512,
					steps=20,
					hires_fix=HiresFixConfig(
						upscale_factor=2.0,
						upscaler=upscaler_type,
						denoising_strength=0.7,
						steps=15,
					),
				)

				result = processor.apply(config, mock_pipe, torch_generator, sample_images)

				mock_pipe.assert_not_called(), f'{upscaler_type} should skip refinement'
				assert result == upscaled_images
