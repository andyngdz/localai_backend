"""Tests for base_generator module."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, Mock, PropertyMock, patch

import pytest

from app.schemas.generators import GeneratorConfig, OutputType
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
		mock_output.images = Mock()
		mock_latent_decoder.decode_latents.return_value = [Mock()]
		mock_latent_decoder.run_safety_checker.return_value = ([Mock()], [False])
		mock_executor.submit = Mock(return_value=AsyncMock(return_value=mock_output)())

		# Execute
		with patch('asyncio.get_event_loop') as mock_loop:
			mock_loop.return_value.run_in_executor = AsyncMock(return_value=mock_output)
			await generator.execute_pipeline(sample_config, 'positive', 'negative')

		# Verify
		mock_model_manager.set_sampler.assert_called_once_with(sample_config.sampler)

	@pytest.mark.asyncio
	@patch('app.features.generators.base_generator.latent_decoder')
	@patch('app.features.generators.base_generator.torch.Generator')
	@patch('app.features.generators.base_generator.seed_manager')
	@patch('app.features.generators.base_generator.progress_callback')
	@patch('app.features.generators.base_generator.model_manager')
	async def test_gets_seed_from_config(
		self,
		mock_model_manager,
		mock_progress_callback,
		mock_seed_manager,
		mock_torch_generator,
		mock_latent_decoder,
		sample_config,
		mock_executor,
	):
		"""Test that seed is retrieved from seed_manager."""
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

		# Verify
		mock_seed_manager.get_seed.assert_called_once_with(sample_config.seed)

	@pytest.mark.asyncio
	@patch('app.features.generators.base_generator.latent_decoder')
	@patch('app.features.generators.base_generator.torch.Generator')
	@patch('app.features.generators.base_generator.seed_manager')
	@patch('app.features.generators.base_generator.progress_callback')
	@patch('app.features.generators.base_generator.model_manager')
	async def test_creates_pipeline_params_correctly(
		self,
		mock_model_manager,
		mock_progress_callback,
		mock_seed_manager,
		mock_torch_generator,
		mock_latent_decoder,
		sample_config,
		mock_executor,
	):
		"""Test that pipeline parameters are created correctly."""
		from app.features.generators.base_generator import BaseGenerator

		# Setup
		mock_pipe = Mock()
		mock_pipe.device = 'cuda'
		mock_model_manager.pipe = mock_pipe
		mock_seed_manager.get_seed.return_value = 12345

		mock_generator_instance = Mock()
		mock_torch_generator.return_value.manual_seed.return_value = mock_generator_instance

		generator = BaseGenerator(mock_executor)

		mock_output = Mock()
		mock_output.images = Mock()
		mock_latent_decoder.decode_latents.return_value = [Mock()]
		mock_latent_decoder.run_safety_checker.return_value = ([Mock()], [False])
		captured_params = {}

		def capture_params(**kwargs):
			captured_params.update(kwargs)
			return mock_output

		mock_pipe.side_effect = capture_params

		# Execute
		with patch('asyncio.get_event_loop') as mock_loop:

			async def mock_executor_func(executor, func):
				return func()

			mock_loop.return_value.run_in_executor = mock_executor_func
			await generator.execute_pipeline(sample_config, 'final positive', 'final negative')

		# Verify pipeline params
		assert captured_params['prompt'] == 'final positive'
		assert captured_params['negative_prompt'] == 'final negative'
		assert captured_params['num_inference_steps'] == 20
		assert captured_params['guidance_scale'] == 7.5
		assert captured_params['height'] == 512
		assert captured_params['width'] == 512
		assert captured_params['num_images_per_prompt'] == 2
		assert captured_params['clip_skip'] == 1
		assert captured_params['output_type'] == OutputType.LATENT
		assert captured_params['generator'] == mock_generator_instance

	@pytest.mark.asyncio
	@patch('app.features.generators.base_generator.latent_decoder')
	@patch('app.features.generators.base_generator.torch.Generator')
	@patch('app.features.generators.base_generator.logger')
	@patch('app.features.generators.base_generator.seed_manager')
	@patch('app.features.generators.base_generator.progress_callback')
	@patch('app.features.generators.base_generator.model_manager')
	async def test_logs_generation_start_and_complete(
		self,
		mock_model_manager,
		mock_progress_callback,
		mock_seed_manager,
		mock_logger,
		mock_torch_generator,
		mock_latent_decoder,
		sample_config,
		mock_executor,
	):
		"""Test that generation start and completion are logged."""
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

		# Verify logging
		assert mock_logger.info.call_count >= 3  # Start, thread start, completion
		log_messages = [str(call) for call in mock_logger.info.call_args_list]
		assert any('Generating image(s)' in msg for msg in log_messages)
		assert any('Starting image generation' in msg for msg in log_messages)
		assert any('completed successfully' in msg for msg in log_messages)

	@pytest.mark.asyncio
	@patch('app.features.generators.base_generator.latent_decoder')
	@patch('app.features.generators.base_generator.torch.Generator')
	@patch('app.features.generators.base_generator.seed_manager')
	@patch('app.features.generators.base_generator.progress_callback')
	@patch('app.features.generators.base_generator.model_manager')
	async def test_returns_pipeline_output(
		self,
		mock_model_manager,
		mock_progress_callback,
		mock_seed_manager,
		mock_torch_generator,
		mock_latent_decoder,
		sample_config,
		mock_executor,
	):
		"""Test that method returns pipeline output with images and NSFW flags."""
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
		mock_images = [Mock(), Mock()]
		mock_latent_decoder.decode_latents.return_value = mock_images
		mock_latent_decoder.run_safety_checker.return_value = (mock_images, [False, False])

		# Execute
		with patch('asyncio.get_event_loop') as mock_loop:
			mock_loop.return_value.run_in_executor = AsyncMock(return_value=mock_output)
			result = await generator.execute_pipeline(sample_config, 'positive', 'negative')

		# Verify result has images and nsfw flags
		assert result.images == mock_images
		assert result.nsfw_content_detected == [False, False]


class TestHiresFixIntegration:
	"""Test hires fix integration in execute_pipeline()."""

	@pytest.mark.asyncio
	@patch('app.features.generators.base_generator.hires_fix_processor')
	@patch('app.features.generators.base_generator.latent_decoder')
	@patch('app.features.generators.base_generator.torch.Generator')
	@patch('app.features.generators.base_generator.seed_manager')
	@patch('app.features.generators.base_generator.progress_callback')
	@patch('app.features.generators.base_generator.model_manager')
	async def test_applies_hires_fix_when_configured(
		self,
		mock_model_manager,
		mock_progress_callback,
		mock_seed_manager,
		mock_torch_generator,
		mock_latent_decoder,
		mock_hires_fix_processor,
		mock_executor,
	):
		"""Test that hires fix is applied when configured."""
		from app.features.generators.base_generator import BaseGenerator

		# Setup
		config = GeneratorConfig(
			prompt='test prompt',
			width=512,
			height=512,
			steps=20,
			hires_fix=HiresFixConfig(
				upscale_factor=2.0,
				upscaler=UpscalerType.LATENT,
				denoising_strength=0.7,
				steps=15,
			),
		)

		mock_pipe = Mock()
		mock_pipe.device = 'cuda'
		mock_model_manager.pipe = mock_pipe
		mock_seed_manager.get_seed.return_value = 12345

		mock_generator_instance = Mock()
		mock_torch_generator.return_value.manual_seed.return_value = mock_generator_instance

		generator = BaseGenerator(mock_executor)

		mock_output = Mock()
		mock_output.images = Mock()
		mock_latent_decoder.decode_latents.return_value = [Mock()]
		mock_latent_decoder.run_safety_checker.return_value = ([Mock()], [False])
		mock_hires_fix_processor.apply.return_value = Mock()

		# Execute
		with patch('asyncio.get_event_loop') as mock_loop:

			async def mock_executor_func(executor, func):
				# Execute the function and return result
				return func()

			mock_loop.return_value.run_in_executor = mock_executor_func
			await generator.execute_pipeline(config, 'positive', 'negative')

		# Verify hires fix was called (it gets called once in the lambda)
		mock_hires_fix_processor.apply.assert_called_once()
		call_args = mock_hires_fix_processor.apply.call_args[0]
		assert call_args[0] == config
		assert call_args[1] == mock_pipe
		assert call_args[3] == mock_generator_instance

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
				upscaler=UpscalerType.LATENT,
				denoising_strength=0.7,
				steps=15,
			),
		)

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
		mock_hires_fix_processor.apply.return_value = Mock()

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
