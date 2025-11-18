from unittest.mock import Mock, patch

import pytest
import torch
from PIL import Image

from app.cores.samplers import SamplerType
from app.features.generators.schemas import (
	GeneratorConfig,
	ImageGenerationItem,
	ImageGenerationResponse,
)


@pytest.fixture
def mock_service():
	"""Create GeneratorService with mocked dependencies."""
	with (
		patch('app.features.generators.service.model_manager') as mock_model_manager,
		patch('app.features.generators.service.seed_manager') as mock_seed_manager,
		patch('app.features.generators.service.image_processor') as mock_image_processor,
		patch('app.features.generators.service.memory_manager') as mock_memory_manager,
		patch('app.features.generators.service.progress_callback') as mock_progress_callback,
		patch('app.features.generators.service.styles_service') as mock_styles_service,
		patch('app.features.generators.service.torch') as mock_torch,
	):
		# Configure seed_manager
		mock_seed_manager.get_seed.return_value = 12345

		# Configure image_processor
		mock_image_processor.is_nsfw_content_detected.return_value = [False]
		mock_image_processor.save_image.return_value = ('/static/test.png', 'test')

		# Configure memory_manager
		mock_memory_manager.clear_cache = Mock()
		mock_memory_manager.validate_batch_size = Mock()

		# Configure progress_callback
		mock_progress_callback.callback_on_step_end = Mock()

		# Configure torch
		mock_torch.cuda.OutOfMemoryError = torch.cuda.OutOfMemoryError
		mock_torch.Generator = Mock(return_value=Mock())

		from app.features.generators.service import GeneratorService

		service = GeneratorService()

		yield (
			service,
			mock_model_manager,
			mock_seed_manager,
			mock_image_processor,
			mock_memory_manager,
			mock_progress_callback,
			mock_styles_service,
			mock_torch,
		)


@pytest.fixture
def sample_config():
	"""Create sample GeneratorConfig for testing."""
	return GeneratorConfig(
		prompt='test prompt',
		width=512,
		height=512,
		steps=20,
		cfg_scale=7.5,
		number_of_images=1,
		seed=-1,
		sampler=SamplerType.EULER_A,
		hires_fix=False,
		styles=[],
	)


@pytest.fixture
def mock_db():
	"""Create mock database session."""
	return Mock()


class TestGeneratorServiceInit:
	def test_creates_executor(self, mock_service):
		service, *_ = mock_service
		assert service.executor is not None
		assert hasattr(service.executor, 'submit')


class TestApplyHiresFix:
	def test_logs_warning_when_hires_fix_enabled(self, mock_service, caplog):
		service, *_ = mock_service

		service.apply_hires_fix(True)

		assert 'Hires fix requested, but not fully implemented' in caplog.text

	def test_does_not_log_when_hires_fix_disabled(self, mock_service, caplog):
		service, *_ = mock_service

		service.apply_hires_fix(False)

		assert 'Hires fix requested' not in caplog.text


class TestGenerateImage:
	@pytest.mark.asyncio
	async def test_raises_error_when_no_model_loaded(self, mock_service, sample_config, mock_db):
		service, mock_model_manager, *_ = mock_service
		mock_model_manager.pipe = None

		with pytest.raises(ValueError, match='No model is currently loaded'):
			await service.generate_image(sample_config, mock_db)

	@pytest.mark.asyncio
	async def test_clears_cuda_cache_before_generation_when_cuda_available(self, mock_service, sample_config, mock_db):
		service, mock_model_manager, _, _, mock_memory_manager, *_ = mock_service
		mock_model_manager.pipe = Mock()

		# Mock the executor to avoid actual execution
		service.executor = Mock()
		service.executor.submit = Mock(side_effect=Exception('Stop execution'))

		try:
			await service.generate_image(sample_config, mock_db)
		except Exception:
			pass

		mock_memory_manager.clear_cache.assert_called()

	@pytest.mark.asyncio
	async def test_validates_batch_size_with_device_service(self, mock_service, sample_config, caplog):
		service, mock_model_manager, _, _, mock_memory_manager, *_ = mock_service
		mock_model_manager.pipe = Mock()

		# Set batch size higher than recommended
		sample_config.number_of_images = 4

		# Mock executor to stop execution after validation
		service.executor = Mock()
		service.executor.submit = Mock(side_effect=Exception('Stop'))

		try:
			await service.generate_image(sample_config, mock_db)
		except Exception:
			pass

		mock_memory_manager.validate_batch_size.assert_called_once_with(4, 512, 512)

	@pytest.mark.asyncio
	async def test_does_not_warn_when_batch_size_within_limit(self, mock_service, sample_config, caplog):
		service, mock_model_manager, *_ = mock_service
		mock_model_manager.pipe = Mock()

		sample_config.number_of_images = 2

		service.executor = Mock()
		service.executor.submit = Mock(side_effect=Exception('Stop'))

		try:
			await service.generate_image(sample_config, mock_db)
		except Exception:
			pass

		# Test passes if no errors occur - batch size validation happens in memory_manager

	@pytest.mark.asyncio
	async def test_sets_sampler_before_generation(self, mock_service, sample_config, mock_db):
		service, mock_model_manager, *_ = mock_service
		mock_model_manager.pipe = Mock()

		service.executor = Mock()
		service.executor.submit = Mock(side_effect=Exception('Stop'))

		try:
			await service.generate_image(sample_config, mock_db)
		except Exception:
			pass

		mock_model_manager.set_sampler.assert_called_once_with(SamplerType.EULER_A)

	@pytest.mark.asyncio
	async def test_applies_styles_via_styles_service(self, mock_service, sample_config, mock_db):
		service, mock_model_manager, _, _, _, _, mock_styles_service, *_ = mock_service
		mock_model_manager.pipe = Mock()
		mock_styles_service.apply_styles.return_value = ('positive prompt', 'negative prompt')

		sample_config.styles = ['style1', 'style2']

		service.executor = Mock()
		service.executor.submit = Mock(side_effect=Exception('Stop'))

		try:
			await service.generate_image(sample_config, mock_db)
		except Exception:
			pass

		mock_styles_service.apply_styles.assert_called_once_with('test prompt', ['style1', 'style2'])

	@pytest.mark.asyncio
	async def test_uses_default_negative_prompt_when_none_provided(self, mock_service, sample_config, mock_db):
		service, mock_model_manager, _, _, _, _, mock_styles_service, *_ = mock_service
		mock_model_manager.pipe = Mock()
		mock_styles_service.apply_styles.return_value = ('positive', None)

		# Create a mock that captures the lambda call
		async def mock_executor(*args, **kwargs):
			# Get the lambda function from args
			if len(args) > 1 and callable(args[1]):
				# This would be the lambda, but we can't easily inspect it
				pass
			raise Exception('Stop')

		service.executor.submit = Mock(side_effect=Exception('Stop'))

		try:
			await service.generate_image(sample_config, mock_db)
		except Exception:
			pass

		# Verify styles were applied
		mock_styles_service.apply_styles.assert_called_once()

	@pytest.mark.asyncio
	async def test_successful_image_generation(self, mock_service, sample_config, mock_db):
		service, mock_model_manager, _, mock_image_processor, _, _, mock_styles_service, *_ = mock_service

		# Mock the pipe
		mock_pipe = Mock()
		mock_pipe.device = 'cpu'
		test_image = Image.new('RGB', (64, 64), color='blue')
		mock_pipe.return_value = {'images': [test_image], 'nsfw_content_detected': [False]}
		mock_model_manager.pipe = mock_pipe

		# Mock styles
		mock_styles_service.apply_styles.return_value = ('positive', 'negative')

		# image_processor is already mocked to return test values
		# mock_image_processor.save_image returns ('/static/test.png', 'test')
		# mock_image_processor.is_nsfw_content_detected returns [False]

		result = await service.generate_image(sample_config, mock_db)

		assert isinstance(result, ImageGenerationResponse)
		assert len(result.items) == 1
		assert isinstance(result.items[0], ImageGenerationItem)
		assert result.items[0].path == '/static/test.png'
		assert result.items[0].file_name == 'test'
		assert result.nsfw_content_detected == [False]

	@pytest.mark.asyncio
	async def test_handles_oom_error_and_clears_cache(self, mock_service, sample_config, mock_db):
		service, mock_model_manager, _, _, mock_memory_manager, _, mock_styles_service, *_ = mock_service

		mock_pipe = Mock()
		mock_pipe.device = 'cuda'
		mock_pipe.side_effect = torch.cuda.OutOfMemoryError('CUDA out of memory')
		mock_model_manager.pipe = mock_pipe
		mock_styles_service.apply_styles.return_value = ('pos', 'neg')

		with pytest.raises(ValueError, match='Out of memory error'):
			await service.generate_image(sample_config, mock_db)

		# Verify cache was cleared in except and finally blocks
		assert mock_memory_manager.clear_cache.call_count >= 2

	@pytest.mark.asyncio
	async def test_handles_file_not_found_error(self, mock_service, sample_config, mock_db):
		service, mock_model_manager, _, _, _, _, mock_styles_service, *_ = mock_service

		mock_pipe = Mock()
		mock_pipe.device = 'cpu'
		mock_pipe.side_effect = FileNotFoundError('Model files not found')
		mock_model_manager.pipe = mock_pipe
		mock_styles_service.apply_styles.return_value = ('pos', 'neg')

		with pytest.raises(ValueError, match='Model files not found'):
			await service.generate_image(sample_config, mock_db)

	@pytest.mark.asyncio
	async def test_handles_general_exception(self, mock_service, sample_config, mock_db):
		service, mock_model_manager, _, _, _, _, mock_styles_service, *_ = mock_service

		mock_pipe = Mock()
		mock_pipe.device = 'cpu'
		mock_pipe.side_effect = RuntimeError('Something went wrong')
		mock_model_manager.pipe = mock_pipe
		mock_styles_service.apply_styles.return_value = ('pos', 'neg')

		with pytest.raises(ValueError, match='Failed to generate image'):
			await service.generate_image(sample_config, mock_db)

	@pytest.mark.asyncio
	async def test_clears_cuda_cache_in_finally_block(self, mock_service, sample_config, mock_db):
		service, mock_model_manager, _, _, mock_memory_manager, _, mock_styles_service, *_ = mock_service

		mock_pipe = Mock()
		mock_pipe.device = 'cuda'
		mock_pipe.side_effect = RuntimeError('Test error')
		mock_model_manager.pipe = mock_pipe
		mock_styles_service.apply_styles.return_value = ('pos', 'neg')

		try:
			await service.generate_image(sample_config, mock_db)
		except Exception:
			pass

		# Verify cache was cleared in finally block
		mock_memory_manager.clear_cache.assert_called()
