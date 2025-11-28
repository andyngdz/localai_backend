"""Tests for base_generator module."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, Mock, PropertyMock, patch

import pytest
import torch
from PIL import Image

from app.schemas.generators import GeneratorConfig
from app.schemas.hires_fix import HiresFixConfig, UpscalerType


@pytest.fixture
def sample_config():
	"""Create a sample generator config."""
	return GeneratorConfig(
		prompt='test prompt',
		negative_prompt='bad quality',
		width=512,
		height=512,
		number_of_images=2,
		steps=20,
		cfg_scale=7.5,
		seed=42,
		clip_skip=1,
	)


@pytest.fixture
def mock_executor():
	"""Create a mock executor."""
	return Mock(spec=ThreadPoolExecutor)


class TestExecutePipeline:
	"""Test execute_pipeline() method."""

	@pytest.mark.asyncio
	@patch('app.features.generators.base_generator.seed_manager')
	@patch('app.features.generators.base_generator.progress_callback')
	@patch('app.features.generators.base_generator.model_manager')
	async def test_validates_model_is_loaded(
		self, mock_model_manager, mock_progress_callback, mock_seed_manager, sample_config, mock_executor
	):
		"""Test that ValueError is raised if model is None."""
		from app.features.generators.base_generator import BaseGenerator

		# Setup - Configure pipe property to raise ValueError when accessed
		type(mock_model_manager).pipe = PropertyMock(side_effect=ValueError('No model is currently loaded'))
		generator = BaseGenerator(mock_executor)

		# Execute & Verify
		with pytest.raises(ValueError, match='No model is currently loaded'):
			await generator.execute_pipeline(sample_config, 'positive', 'negative')

	@pytest.mark.asyncio
	@patch('app.features.generators.base_generator.latent_decoder')
	@patch('app.features.generators.base_generator.torch.Generator')
	@patch('app.features.generators.base_generator.seed_manager')
	@patch('app.features.generators.base_generator.progress_callback')
	@patch('app.features.generators.base_generator.model_manager')
	async def test_sets_sampler(
		self,
		mock_model_manager,
		mock_progress_callback,
		mock_seed_manager,
		mock_torch_generator,
		mock_latent_decoder,
		sample_config,
		mock_executor,
	):
		"""Test that sampler is set before generation."""
		from app.features.generators.base_generator import BaseGenerator

		# Setup
		mock_pipe = Mock()
		mock_pipe.device = 'cuda'
		mock_model_manager.pipe = mock_pipe
		mock_seed_manager.get_seed.return_value = 12345
		mock_torch_generator.return_value.manual_seed.return_value = Mock()
		generator = BaseGenerator(mock_executor)

		# Mock executor to return immediately
		mock_output = Mock()
		mock_output.images = torch.randn(1, 4, 64, 64)
		mock_pipe.return_value = mock_output
		mock_latent_decoder.decode_latents.return_value = [Mock()]
		mock_latent_decoder.run_safety_checker.return_value = ([Mock()], [False])
		mock_executor.submit = Mock(return_value=AsyncMock(return_value=mock_output)())

		# Execute
		with patch('asyncio.get_event_loop') as mock_loop:
			mock_loop.return_value.run_in_executor = AsyncMock(return_value=mock_output)
			await generator.execute_pipeline(sample_config, 'positive', 'negative')

		# Verify sampler was set
		mock_model_manager.set_sampler.assert_called_once_with(sample_config.sampler)

	@pytest.mark.asyncio
	@patch('app.features.generators.base_generator.hires_fix_processor')
	@patch('app.features.generators.base_generator.latent_decoder')
	@patch('app.features.generators.base_generator.torch.Generator')
	@patch('app.features.generators.base_generator.seed_manager')
	@patch('app.features.generators.base_generator.progress_callback')
	@patch('app.features.generators.base_generator.model_manager')
	async def test_skips_hires_fix_when_not_configured(
		self,
		mock_model_manager,
		mock_progress_callback,
		mock_seed_manager,
		mock_torch_generator,
		mock_latent_decoder,
		mock_hires_fix_processor,
		sample_config,
		mock_executor,
	):
		"""Test that hires fix is skipped when not configured."""
		from app.features.generators.base_generator import BaseGenerator

		# Setup
		mock_pipe = Mock()
		mock_pipe.device = 'cuda'
		mock_model_manager.pipe = mock_pipe
		mock_seed_manager.get_seed.return_value = 12345
		mock_torch_generator.return_value.manual_seed.return_value = Mock()
		generator = BaseGenerator(mock_executor)

		mock_output = Mock()
		mock_output.images = Mock()
		mock_latent_decoder.decode_latents.return_value = [Mock()]
		mock_latent_decoder.run_safety_checker.return_value = ([Mock()], [False])

		# Execute
		with patch('asyncio.get_event_loop') as mock_loop:
			mock_loop.return_value.run_in_executor = AsyncMock(return_value=mock_output)
			await generator.execute_pipeline(sample_config, 'positive', 'negative')

		# Verify hires fix was NOT called
		mock_hires_fix_processor.apply.assert_not_called()

	@pytest.mark.asyncio
	@patch('app.features.generators.base_generator.hires_fix_processor')
	@patch('app.features.generators.base_generator.latent_decoder')
	@patch('app.features.generators.base_generator.torch.Generator')
	@patch('app.features.generators.base_generator.logger')
	@patch('app.features.generators.base_generator.seed_manager')
	@patch('app.features.generators.base_generator.progress_callback')
	@patch('app.features.generators.base_generator.model_manager')
	async def test_logs_hires_fix_application(
		self,
		mock_model_manager,
		mock_progress_callback,
		mock_seed_manager,
		mock_logger,
		mock_torch_generator,
		mock_latent_decoder,
		mock_hires_fix_processor,
		mock_executor,
	):
		"""Test that hires fix application is logged."""
		from app.features.generators.base_generator import BaseGenerator

		# Setup
		config = GeneratorConfig(
			prompt='test prompt',
			width=512,
			height=512,
			steps=20,
			hires_fix=HiresFixConfig(
				upscale_factor=2.0,
				upscaler=UpscalerType.LANCZOS,
				denoising_strength=0.7,
				steps=15,
			),
		)

		# Create mock output with real tensor for latents
		mock_output = Mock()
		mock_latents = torch.randn(1, 4, 64, 64)
		mock_output.images = mock_latents

		mock_pipe = Mock()
		mock_pipe.device = 'cuda'
		mock_pipe.return_value = mock_output  # Make pipe() return mock_output
		mock_model_manager.pipe = mock_pipe
		mock_seed_manager.get_seed.return_value = 12345
		mock_torch_generator.return_value.manual_seed.return_value = Mock()
		generator = BaseGenerator(mock_executor)

		mock_base_images = [Image.new('RGB', (512, 512))]
		mock_latent_decoder.decode_latents.return_value = mock_base_images
		mock_latent_decoder.run_safety_checker.return_value = (mock_base_images, [False])
		mock_hires_fix_processor.apply.return_value = [Image.new('RGB', (1024, 1024))]

		# Execute
		with patch('asyncio.get_event_loop') as mock_loop:

			async def mock_executor_func(executor, func):
				# Execute the function and return result
				return func()

			mock_loop.return_value.run_in_executor = mock_executor_func
			await generator.execute_pipeline(config, 'positive', 'negative')

		# Verify logging
		log_messages = [str(call) for call in mock_logger.info.call_args_list]
		assert any('Applying hires fix' in msg for msg in log_messages)


class TestApplyHiresFixToSafeImages:
	"""Test _apply_hires_fix_to_safe_images() method."""

	@pytest.mark.asyncio
	@patch('app.features.generators.base_generator.hires_fix_processor')
	@patch('app.features.generators.base_generator.latent_decoder')
	@patch('app.features.generators.base_generator.torch.Generator')
	@patch('app.features.generators.base_generator.logger')
	@patch('app.features.generators.base_generator.seed_manager')
	@patch('app.features.generators.base_generator.progress_callback')
	@patch('app.features.generators.base_generator.model_manager')
	async def test_skips_hires_fix_when_all_images_nsfw(
		self,
		mock_model_manager,
		mock_progress_callback,
		mock_seed_manager,
		mock_logger,
		mock_torch_generator,
		mock_latent_decoder,
		mock_hires_fix_processor,
		mock_executor,
	):
		"""Test that hires fix is skipped when all images are flagged as NSFW."""
		from app.features.generators.base_generator import BaseGenerator

		# Setup
		config = GeneratorConfig(
			prompt='test prompt',
			width=512,
			height=512,
			steps=20,
			hires_fix=HiresFixConfig(
				upscale_factor=2.0,
				upscaler=UpscalerType.LANCZOS,
				denoising_strength=0.7,
				steps=15,
			),
		)

		# Create mock output with real tensor for latents
		mock_output = Mock()
		mock_latents = torch.randn(1, 4, 64, 64)
		mock_output.images = mock_latents

		mock_pipe = Mock()
		mock_pipe.device = 'cuda'
		mock_pipe.return_value = mock_output
		mock_model_manager.pipe = mock_pipe
		mock_seed_manager.get_seed.return_value = 12345
		mock_torch_generator.return_value.manual_seed.return_value = Mock()
		generator = BaseGenerator(mock_executor)

		mock_base_images = [Image.new('RGB', (512, 512))]
		mock_latent_decoder.decode_latents.return_value = mock_base_images
		# All images flagged as NSFW
		mock_latent_decoder.run_safety_checker.return_value = (mock_base_images, [True])

		# Execute
		with patch('asyncio.get_event_loop') as mock_loop:

			async def mock_executor_func(executor, func):
				return func()

			mock_loop.return_value.run_in_executor = mock_executor_func
			await generator.execute_pipeline(config, 'positive', 'negative')

		# Verify hires fix was NOT called and warning was logged
		mock_hires_fix_processor.apply.assert_not_called()
		log_messages = [str(call) for call in mock_logger.warning.call_args_list]
		assert any('All images flagged as NSFW' in msg for msg in log_messages)


class TestBaseGeneratorInit:
	"""Test BaseGenerator initialization."""

	def test_initializes_with_executor(self, mock_executor):
		"""Test that generator initializes with executor."""
		from app.features.generators.base_generator import BaseGenerator

		generator = BaseGenerator(mock_executor)

		assert generator.executor == mock_executor

	def test_executor_is_stored(self):
		"""Test that executor is properly stored."""
		from app.features.generators.base_generator import BaseGenerator

		executor = ThreadPoolExecutor(max_workers=1)
		generator = BaseGenerator(executor)

		assert isinstance(generator.executor, ThreadPoolExecutor)
		executor.shutdown(wait=False)
