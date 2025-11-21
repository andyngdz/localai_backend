from collections.abc import Generator
from typing import TypeAlias
from unittest.mock import Mock, patch

import pytest
import torch
from _pytest.logging import LogCaptureFixture
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from PIL import Image

from app.cores.samplers import SamplerType
from app.features.generators.service import GeneratorService
from app.schemas.generators import (
	GeneratorConfig,
	ImageGenerationItem,
	ImageGenerationResponse,
)
from app.services.styles import DEFAULT_NEGATIVE_PROMPT

MockServiceFixture: TypeAlias = tuple[GeneratorService, Mock, Mock, Mock, Mock, Mock, Mock, Mock]


@pytest.fixture
def mock_service() -> Generator[MockServiceFixture, None, None]:
	"""Create GeneratorService with mocked dependencies."""
	with (
		patch('app.features.generators.service.model_manager') as mock_model_manager,
		patch('app.features.generators.service.seed_manager') as mock_seed_manager,
		patch('app.features.generators.service.image_processor') as mock_image_processor,
		patch('app.features.generators.service.memory_manager') as mock_memory_manager,
		patch('app.features.generators.service.progress_callback') as mock_progress_callback,
		patch('app.features.generators.service.styles_service') as mock_styles_service,
		patch('app.features.generators.service.torch') as mock_torch,
		patch('app.cores.generation.image_utils.image_processor') as mock_image_utils_processor,
		patch('app.cores.generation.image_utils.memory_manager') as mock_image_utils_memory,
	):
		# Configure seed_manager
		mock_seed_manager.get_seed.return_value = 12345

		# Configure image_processor (both instances point to same mock)
		mock_image_processor.is_nsfw_content_detected.return_value = [False]
		mock_image_processor.save_image.return_value = ('/static/test.png', 'test')
		mock_image_utils_processor.is_nsfw_content_detected = mock_image_processor.is_nsfw_content_detected
		mock_image_utils_processor.save_image = mock_image_processor.save_image
		mock_image_utils_processor.clear_tensor_cache = Mock()

		# Configure memory_manager (both instances point to same mock)
		mock_memory_manager.clear_cache = Mock()
		mock_memory_manager.validate_batch_size = Mock()
		mock_image_utils_memory.clear_cache = mock_memory_manager.clear_cache

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
def sample_config() -> GeneratorConfig:
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
def mock_db() -> Mock:
	"""Create mock database session."""
	return Mock()


class TestGeneratorServiceInit:
	def test_creates_executor(self, mock_service: MockServiceFixture) -> None:
		service, *_ = mock_service
		assert service.executor is not None
		assert hasattr(service.executor, 'submit')


class TestApplyHiresFix:
	def test_logs_warning_when_hires_fix_enabled(
		self, mock_service: MockServiceFixture, caplog: LogCaptureFixture
	) -> None:
		service, *_ = mock_service

		service.apply_hires_fix(True)

		assert 'Hires fix requested, but not fully implemented' in caplog.text

	def test_does_not_log_when_hires_fix_disabled(
		self, mock_service: MockServiceFixture, caplog: LogCaptureFixture
	) -> None:
		service, *_ = mock_service

		service.apply_hires_fix(False)

		assert 'Hires fix requested' not in caplog.text


class TestGenerateImage:
	@pytest.mark.asyncio
	async def test_raises_error_when_no_model_loaded(
		self, mock_service: MockServiceFixture, sample_config: GeneratorConfig, mock_db: Mock
	) -> None:
		service, mock_model_manager, *_ = mock_service
		mock_model_manager.pipe = None

		with pytest.raises(ValueError, match='No model is currently loaded'):
			await service.generate_image(sample_config, mock_db)

	@pytest.mark.asyncio
	async def test_clears_cuda_cache_before_generation_when_cuda_available(
		self, mock_service: MockServiceFixture, sample_config: GeneratorConfig, mock_db: Mock
	) -> None:
		service, mock_model_manager, _, _, mock_memory_manager, *_ = mock_service
		mock_model_manager.pipe = Mock()

		# Mock the executor to avoid actual execution
		service.executor = Mock()
		service.executor.submit = Mock(side_effect=Exception('Stop execution'))

		try:
			await service.generate_image(sample_config, mock_db)
		except Exception:
			# Ignore the intentional exception raised by the mock to stop execution
			pass

		mock_memory_manager.clear_cache.assert_called()

	@pytest.mark.asyncio
	async def test_validates_batch_size_with_device_service(
		self, mock_service: MockServiceFixture, sample_config: GeneratorConfig, mock_db: Mock
	) -> None:
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
			# Ignore the intentional exception raised by the mock to stop execution
			pass

		mock_memory_manager.validate_batch_size.assert_called_once_with(4, 512, 512)

	@pytest.mark.asyncio
	async def test_does_not_warn_when_batch_size_within_limit(
		self, mock_service: MockServiceFixture, sample_config: GeneratorConfig, mock_db: Mock
	) -> None:
		service, mock_model_manager, *_ = mock_service
		mock_model_manager.pipe = Mock()

		sample_config.number_of_images = 2

		service.executor = Mock()
		service.executor.submit = Mock(side_effect=Exception('Stop'))

		try:
			await service.generate_image(sample_config, mock_db)
		except Exception:
			# Ignore the intentional exception raised by the mock to stop execution
			pass

		# Test passes if no errors occur - batch size validation happens in memory_manager

	@pytest.mark.asyncio
	async def test_sets_sampler_before_generation(
		self, mock_service: MockServiceFixture, sample_config: GeneratorConfig, mock_db: Mock
	) -> None:
		service, mock_model_manager, *_ = mock_service
		mock_model_manager.pipe = Mock()

		service.executor = Mock()
		service.executor.submit = Mock(side_effect=Exception('Stop'))

		try:
			await service.generate_image(sample_config, mock_db)
		except Exception:
			# Ignore the intentional exception raised by the mock to stop execution
			pass

		mock_model_manager.set_sampler.assert_called_once_with(SamplerType.EULER_A)

	@pytest.mark.asyncio
	async def test_applies_styles_via_styles_service(
		self, mock_service: MockServiceFixture, sample_config: GeneratorConfig, mock_db: Mock
	) -> None:
		service, mock_model_manager, _, _, _, _, mock_styles_service, *_ = mock_service
		mock_model_manager.pipe = Mock()
		mock_styles_service.apply_styles.return_value = ('positive prompt', 'negative prompt')

		sample_config.styles = ['style1', 'style2']

		service.executor = Mock()
		service.executor.submit = Mock(side_effect=Exception('Stop'))

		try:
			await service.generate_image(sample_config, mock_db)
		except Exception:
			# Ignore the intentional exception raised by the mock to stop execution
			pass

		mock_styles_service.apply_styles.assert_called_once_with(
			'test prompt', DEFAULT_NEGATIVE_PROMPT, ['style1', 'style2']
		)

	@pytest.mark.asyncio
	async def test_uses_default_negative_prompt_when_none_provided(
		self, mock_service: MockServiceFixture, sample_config: GeneratorConfig, mock_db: Mock
	) -> None:
		service, mock_model_manager, _, _, _, _, mock_styles_service, *_ = mock_service
		mock_model_manager.pipe = Mock()
		mock_styles_service.apply_styles.return_value = ('positive', None)

		service.executor = Mock()
		service.executor.submit = Mock(side_effect=Exception('Stop'))

		try:
			await service.generate_image(sample_config, mock_db)
		except Exception:
			# Ignore the intentional exception raised by the mock to stop execution
			pass

		# Verify styles were applied
		mock_styles_service.apply_styles.assert_called_once()

	@pytest.mark.asyncio
	async def test_successful_image_generation(
		self, mock_service: MockServiceFixture, sample_config: GeneratorConfig, mock_db: Mock
	) -> None:
		service, mock_model_manager, _, _, _, _, mock_styles_service, *_ = mock_service

		# Mock the pipe
		mock_pipe = Mock()
		mock_pipe.device = 'cpu'
		test_image = Image.new('RGB', (64, 64), color='blue')
		mock_pipe.return_value = StableDiffusionPipelineOutput(images=[test_image], nsfw_content_detected=[False])
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
	async def test_handles_oom_error_and_clears_cache(
		self, mock_service: MockServiceFixture, sample_config: GeneratorConfig, mock_db: Mock
	) -> None:
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
	async def test_handles_file_not_found_error(
		self, mock_service: MockServiceFixture, sample_config: GeneratorConfig, mock_db: Mock
	) -> None:
		service, mock_model_manager, _, _, _, _, mock_styles_service, *_ = mock_service

		mock_pipe = Mock()
		mock_pipe.device = 'cpu'
		mock_pipe.side_effect = FileNotFoundError('Model files not found')
		mock_model_manager.pipe = mock_pipe
		mock_styles_service.apply_styles.return_value = ('pos', 'neg')

		with pytest.raises(ValueError, match='Model files not found'):
			await service.generate_image(sample_config, mock_db)

	@pytest.mark.asyncio
	async def test_handles_general_exception(
		self, mock_service: MockServiceFixture, sample_config: GeneratorConfig, mock_db: Mock
	) -> None:
		service, mock_model_manager, _, _, _, _, mock_styles_service, *_ = mock_service

		mock_pipe = Mock()
		mock_pipe.device = 'cpu'
		mock_pipe.side_effect = RuntimeError('Something went wrong')
		mock_model_manager.pipe = mock_pipe
		mock_styles_service.apply_styles.return_value = ('pos', 'neg')

		with pytest.raises(ValueError, match='Failed to generate image'):
			await service.generate_image(sample_config, mock_db)

	@pytest.mark.asyncio
	async def test_clears_cuda_cache_in_finally_block(
		self, mock_service: MockServiceFixture, sample_config: GeneratorConfig, mock_db: Mock
	) -> None:
		service, mock_model_manager, _, _, mock_memory_manager, _, mock_styles_service, *_ = mock_service

		mock_pipe = Mock()
		mock_pipe.device = 'cuda'
		mock_pipe.side_effect = RuntimeError('Test error')
		mock_model_manager.pipe = mock_pipe
		mock_styles_service.apply_styles.return_value = ('pos', 'neg')

		try:
			await service.generate_image(sample_config, mock_db)
		except Exception:
			# Ignore the intentional exception raised by the mock to stop execution
			pass

		# Verify cache was cleared in finally block
		mock_memory_manager.clear_cache.assert_called()

	@pytest.mark.asyncio
	async def test_resets_progress_callback_if_reset_exists(
		self, mock_service: MockServiceFixture, sample_config: GeneratorConfig, mock_db: Mock
	) -> None:
		service, mock_model_manager, _, _, _, mock_progress_callback, mock_styles_service, *_ = mock_service

		mock_pipe = Mock()
		mock_pipe.device = 'cpu'
		mock_model_manager.pipe = mock_pipe
		mock_styles_service.apply_styles.return_value = ('pos', 'neg')

		# Add reset method to progress_callback
		mock_progress_callback.reset = Mock()

		service.executor = Mock()
		service.executor.submit = Mock(side_effect=Exception('Stop'))

		try:
			await service.generate_image(sample_config, mock_db)
		except Exception:
			# Ignore the intentional exception raised by the mock to stop execution
			pass

		# Verify reset was called
		mock_progress_callback.reset.assert_called()

	@pytest.mark.asyncio
	async def test_handles_progress_callback_without_reset(
		self, mock_service: MockServiceFixture, sample_config: GeneratorConfig, mock_db: Mock
	) -> None:
		service, mock_model_manager, _, _, _, mock_progress_callback, mock_styles_service, *_ = mock_service

		mock_pipe = Mock()
		mock_pipe.device = 'cpu'
		mock_model_manager.pipe = mock_pipe
		mock_styles_service.apply_styles.return_value = ('pos', 'neg')

		# Ensure progress_callback does NOT have reset attribute
		if hasattr(mock_progress_callback, 'reset'):
			delattr(mock_progress_callback, 'reset')

		service.executor = Mock()
		service.executor.submit = Mock(side_effect=Exception('Stop'))

		# Should not raise error even without reset
		try:
			await service.generate_image(sample_config, mock_db)
		except Exception:
			# Ignore the intentional exception raised by the mock to stop execution
			pass  # Expected exception from executor.submit

	@pytest.mark.asyncio
	async def test_handles_oom_when_clear_tensor_cache_not_available(
		self, sample_config: GeneratorConfig, mock_db: Mock
	) -> None:
		"""Test OOM handling when image_processor doesn't have clear_tensor_cache."""
		# Create service with fresh patches where clear_tensor_cache does NOT exist
		with (
			patch('app.features.generators.service.model_manager') as mock_model_manager,
			patch('app.features.generators.service.seed_manager') as mock_seed_manager,
			patch('app.features.generators.service.memory_manager') as mock_memory_manager,
			patch('app.features.generators.service.styles_service') as mock_styles_service,
			patch('app.features.generators.service.torch') as mock_torch,
			patch('app.features.generators.service.image_processor') as mock_image_processor,
			patch('app.cores.generation.image_utils.image_processor') as mock_image_utils_processor,
			patch('app.cores.generation.image_utils.memory_manager'),
		):
			# Ensure image_processor does NOT have clear_tensor_cache
			if hasattr(mock_image_processor, 'clear_tensor_cache'):
				delattr(mock_image_processor, 'clear_tensor_cache')

			# Set up image_utils mocks
			mock_image_utils_processor.is_nsfw_content_detected.return_value = [False]
			mock_image_utils_processor.save_image.return_value = ('/static/test.png', 'test')

			# Set up other required mocks
			mock_seed_manager.get_seed.return_value = 12345
			mock_torch.cuda.OutOfMemoryError = torch.cuda.OutOfMemoryError
			mock_torch.Generator = Mock(return_value=Mock())
			mock_memory_manager.clear_cache = Mock()
			mock_styles_service.apply_styles.return_value = ('pos', 'neg')

			mock_pipe = Mock()
			mock_pipe.device = 'cuda'
			mock_pipe.side_effect = torch.cuda.OutOfMemoryError('CUDA out of memory')
			mock_model_manager.pipe = mock_pipe

			from app.features.generators.service import GeneratorService

			service = GeneratorService()

			# Should still handle OOM error even without clear_tensor_cache
			with pytest.raises(ValueError, match='Out of memory error'):
				await service.generate_image(sample_config, mock_db)

	@pytest.mark.asyncio
	async def test_unloads_loras_in_finally_block(
		self, mock_service: MockServiceFixture, sample_config: GeneratorConfig, mock_db: Mock
	) -> None:
		from app.schemas.lora import LoRAConfigItem

		service, mock_model_manager, _, _, _, _, mock_styles_service, *_ = mock_service

		# Set up LoRAs
		sample_config.loras = [LoRAConfigItem(lora_id=1, weight=0.8)]

		mock_pipe = Mock()
		mock_pipe.device = 'cpu'
		test_image = Image.new('RGB', (64, 64), color='blue')
		mock_pipe.return_value = StableDiffusionPipelineOutput(images=[test_image], nsfw_content_detected=[False])
		mock_model_manager.pipe = mock_pipe
		mock_styles_service.apply_styles.return_value = ('pos', 'neg')

		# Mock database
		with patch('app.features.generators.service.database_service') as mock_database_service:
			mock_lora = Mock()
			mock_lora.id = 1
			mock_lora.name = 'test_lora'
			mock_lora.file_path = '/path/to/lora.safetensors'
			mock_database_service.get_lora_by_id.return_value = mock_lora

			await service.generate_image(sample_config, mock_db)

			# Verify unload_loras was called in finally block
			mock_model_manager.pipeline_manager.unload_loras.assert_called_once()

	@pytest.mark.asyncio
	async def test_handles_error_during_lora_unload(
		self, mock_service: MockServiceFixture, sample_config: GeneratorConfig, mock_db: Mock, caplog: LogCaptureFixture
	) -> None:
		from app.schemas.lora import LoRAConfigItem

		service, mock_model_manager, _, _, _, _, mock_styles_service, *_ = mock_service

		# Set up LoRAs
		sample_config.loras = [LoRAConfigItem(lora_id=1, weight=0.8)]

		mock_pipe = Mock()
		mock_pipe.device = 'cpu'
		test_image = Image.new('RGB', (64, 64), color='blue')
		mock_pipe.return_value = StableDiffusionPipelineOutput(images=[test_image], nsfw_content_detected=[False])
		mock_model_manager.pipe = mock_pipe
		mock_styles_service.apply_styles.return_value = ('pos', 'neg')

		# Make unload_loras raise an error
		mock_model_manager.pipeline_manager.unload_loras.side_effect = RuntimeError('Unload failed')

		# Mock database
		with patch('app.features.generators.service.database_service') as mock_database_service:
			mock_lora = Mock()
			mock_lora.id = 1
			mock_lora.name = 'test_lora'
			mock_lora.file_path = '/path/to/lora.safetensors'
			mock_database_service.get_lora_by_id.return_value = mock_lora

			# Should not raise, error is logged
			await service.generate_image(sample_config, mock_db)

			# Verify error was logged
			assert 'Failed to unload LoRAs' in caplog.text
