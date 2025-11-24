"""Tests for GeneratorService after modular refactoring."""

from collections.abc import Generator
from typing import TypeAlias
from unittest.mock import AsyncMock, Mock, patch

import pytest
import torch
from PIL import Image

from app.cores.samplers import SamplerType
from app.features.generators.service import GeneratorService
from app.schemas.generators import GeneratorConfig, ImageGenerationItem, ImageGenerationResponse

MockServiceFixture: TypeAlias = tuple[GeneratorService, Mock, Mock, Mock, Mock, Mock]


@pytest.fixture
def mock_service() -> Generator[MockServiceFixture, None, None]:
	"""Create GeneratorService with mocked module dependencies."""
	with (
		patch('app.features.generators.service.config_validator') as mock_config_validator,
		patch('app.features.generators.service.resource_manager') as mock_resource_manager,
		patch('app.features.generators.service.lora_loader') as mock_lora_loader,
		patch('app.features.generators.service.prompt_processor') as mock_prompt_processor,
		patch('app.features.generators.service.response_builder') as mock_response_builder,
	):
		# Configure mocks
		mock_config_validator.validate_config = Mock()
		mock_resource_manager.prepare_for_generation = Mock()
		mock_resource_manager.cleanup_after_generation = Mock()
		mock_resource_manager.handle_oom_error = Mock()
		mock_lora_loader.load_loras_for_generation = Mock(return_value=False)
		mock_lora_loader.unload_loras = Mock()
		mock_prompt_processor.prepare_prompts = Mock(return_value=('positive', 'negative'))
		mock_response_builder.build_response = Mock(
			return_value=ImageGenerationResponse(
				items=[ImageGenerationItem(path='/static/test.png', file_name='test')],
				nsfw_content_detected=[False],
			)
		)

		service = GeneratorService()

		yield (
			service,
			mock_config_validator,
			mock_resource_manager,
			mock_lora_loader,
			mock_prompt_processor,
			mock_response_builder,
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
	"""Tests for GeneratorService initialization."""

	def test_creates_executor(self, mock_service: MockServiceFixture) -> None:
		"""Test that service creates ThreadPoolExecutor."""
		service, *_ = mock_service
		assert service.executor is not None
		assert hasattr(service.executor, 'submit')

	def test_creates_base_generator(self, mock_service: MockServiceFixture) -> None:
		"""Test that service creates BaseGenerator instance."""
		service, *_ = mock_service
		assert service.generator is not None
		assert hasattr(service.generator, 'execute_pipeline')


class TestGenerateImageOrchestration:
	"""Tests for generate_image orchestration flow."""

	@pytest.mark.asyncio
	async def test_validates_config_before_generation(
		self, mock_service: MockServiceFixture, sample_config: GeneratorConfig, mock_db: Mock
	) -> None:
		"""Test that config validation is called first."""
		service, mock_config_validator, *_ = mock_service

		# Mock generator to avoid actual execution
		service.generator.execute_pipeline = AsyncMock(
			return_value=Mock(images=[Image.new('RGB', (64, 64))], nsfw_content_detected=[False])
		)

		await service.generate_image(sample_config, mock_db)

		mock_config_validator.validate_config.assert_called_once_with(sample_config)

	@pytest.mark.asyncio
	async def test_prepares_resources_before_generation(
		self, mock_service: MockServiceFixture, sample_config: GeneratorConfig, mock_db: Mock
	) -> None:
		"""Test that resource preparation is called."""
		service, _, mock_resource_manager, *_ = mock_service

		service.generator.execute_pipeline = AsyncMock(
			return_value=Mock(images=[Image.new('RGB', (64, 64))], nsfw_content_detected=[False])
		)

		await service.generate_image(sample_config, mock_db)

		mock_resource_manager.prepare_for_generation.assert_called_once()

	@pytest.mark.asyncio
	async def test_loads_loras_when_specified(
		self, mock_service: MockServiceFixture, sample_config: GeneratorConfig, mock_db: Mock
	) -> None:
		"""Test that LoRAs are loaded when configured."""
		service, _, _, mock_lora_loader, *_ = mock_service
		mock_lora_loader.load_loras_for_generation.return_value = True

		service.generator.execute_pipeline = AsyncMock(
			return_value=Mock(images=[Image.new('RGB', (64, 64))], nsfw_content_detected=[False])
		)

		await service.generate_image(sample_config, mock_db)

		mock_lora_loader.load_loras_for_generation.assert_called_once_with(sample_config, mock_db)

	@pytest.mark.asyncio
	async def test_processes_prompts_with_styles(
		self, mock_service: MockServiceFixture, sample_config: GeneratorConfig, mock_db: Mock
	) -> None:
		"""Test that prompts are processed through prompt_processor."""
		service, _, _, _, mock_prompt_processor, _ = mock_service

		service.generator.execute_pipeline = AsyncMock(
			return_value=Mock(images=[Image.new('RGB', (64, 64))], nsfw_content_detected=[False])
		)

		await service.generate_image(sample_config, mock_db)

		mock_prompt_processor.prepare_prompts.assert_called_once_with(sample_config)

	@pytest.mark.asyncio
	async def test_executes_pipeline_with_processed_prompts(
		self, mock_service: MockServiceFixture, sample_config: GeneratorConfig, mock_db: Mock
	) -> None:
		"""Test that pipeline execution receives processed prompts."""
		service, *_, mock_prompt_processor, _ = mock_service
		mock_prompt_processor.prepare_prompts.return_value = ('positive_test', 'negative_test')

		service.generator.execute_pipeline = AsyncMock(
			return_value=Mock(images=[Image.new('RGB', (64, 64))], nsfw_content_detected=[False])
		)

		await service.generate_image(sample_config, mock_db)

		service.generator.execute_pipeline.assert_called_once_with(sample_config, 'positive_test', 'negative_test')

	@pytest.mark.asyncio
	async def test_builds_response_from_output(
		self, mock_service: MockServiceFixture, sample_config: GeneratorConfig, mock_db: Mock
	) -> None:
		"""Test that response is built from pipeline output."""
		service, *_, mock_response_builder = mock_service

		mock_output = Mock(images=[Image.new('RGB', (64, 64))], nsfw_content_detected=[False])
		service.generator.execute_pipeline = AsyncMock(return_value=mock_output)

		await service.generate_image(sample_config, mock_db)

		mock_response_builder.build_response.assert_called_once()

	@pytest.mark.asyncio
	async def test_returns_image_generation_response(
		self, mock_service: MockServiceFixture, sample_config: GeneratorConfig, mock_db: Mock
	) -> None:
		"""Test that method returns ImageGenerationResponse."""
		service, *_ = mock_service

		service.generator.execute_pipeline = AsyncMock(
			return_value=Mock(images=[Image.new('RGB', (64, 64))], nsfw_content_detected=[False])
		)

		result = await service.generate_image(sample_config, mock_db)

		assert isinstance(result, ImageGenerationResponse)
		assert len(result.items) == 1
		assert result.items[0].path == '/static/test.png'


class TestGenerateImageErrorHandling:
	"""Tests for error handling in generate_image."""

	@pytest.mark.asyncio
	async def test_raises_error_when_validation_fails(
		self, mock_service: MockServiceFixture, sample_config: GeneratorConfig, mock_db: Mock
	) -> None:
		"""Test that validation errors are propagated."""
		service, mock_config_validator, *_ = mock_service
		mock_config_validator.validate_config.side_effect = ValueError('No model loaded')

		with pytest.raises(ValueError, match='No model loaded'):
			await service.generate_image(sample_config, mock_db)

	@pytest.mark.asyncio
	async def test_handles_file_not_found_error(
		self, mock_service: MockServiceFixture, sample_config: GeneratorConfig, mock_db: Mock
	) -> None:
		"""Test FileNotFoundError handling."""
		service, *_ = mock_service
		service.generator.execute_pipeline = AsyncMock(side_effect=FileNotFoundError('Model files missing'))

		with pytest.raises(ValueError, match='Model files not found'):
			await service.generate_image(sample_config, mock_db)

	@pytest.mark.asyncio
	async def test_handles_oom_error_and_calls_cleanup(
		self, mock_service: MockServiceFixture, sample_config: GeneratorConfig, mock_db: Mock
	) -> None:
		"""Test OOM error handling."""
		service, _, mock_resource_manager, *_ = mock_service
		service.generator.execute_pipeline = AsyncMock(side_effect=torch.cuda.OutOfMemoryError('CUDA OOM'))

		with pytest.raises(ValueError, match='Out of memory error'):
			await service.generate_image(sample_config, mock_db)

		mock_resource_manager.handle_oom_error.assert_called_once()

	@pytest.mark.asyncio
	async def test_handles_general_exception(
		self, mock_service: MockServiceFixture, sample_config: GeneratorConfig, mock_db: Mock
	) -> None:
		"""Test general exception handling."""
		service, *_ = mock_service
		service.generator.execute_pipeline = AsyncMock(side_effect=RuntimeError('Something went wrong'))

		with pytest.raises(ValueError, match='Failed to generate image'):
			await service.generate_image(sample_config, mock_db)


class TestGenerateImageCleanup:
	"""Tests for resource cleanup in finally block."""

	@pytest.mark.asyncio
	async def test_cleans_up_resources_after_success(
		self, mock_service: MockServiceFixture, sample_config: GeneratorConfig, mock_db: Mock
	) -> None:
		"""Test that cleanup is called after successful generation."""
		service, _, mock_resource_manager, mock_lora_loader, *_ = mock_service
		mock_lora_loader.load_loras_for_generation.return_value = True

		service.generator.execute_pipeline = AsyncMock(
			return_value=Mock(images=[Image.new('RGB', (64, 64))], nsfw_content_detected=[False])
		)

		await service.generate_image(sample_config, mock_db)

		mock_resource_manager.cleanup_after_generation.assert_called_once()
		# Verify it was called with correct arguments
		call_args = mock_resource_manager.cleanup_after_generation.call_args
		assert call_args.kwargs['loras_loaded'] is True
		assert call_args.kwargs['unload_loras_fn'] == mock_lora_loader.unload_loras

	@pytest.mark.asyncio
	async def test_cleans_up_resources_after_error(
		self, mock_service: MockServiceFixture, sample_config: GeneratorConfig, mock_db: Mock
	) -> None:
		"""Test that cleanup is called even after errors."""
		service, _, mock_resource_manager, *_ = mock_service
		service.generator.execute_pipeline = AsyncMock(side_effect=RuntimeError('Test error'))

		try:
			await service.generate_image(sample_config, mock_db)
		except ValueError:
			pass  # Expected

		mock_resource_manager.cleanup_after_generation.assert_called_once()

	@pytest.mark.asyncio
	async def test_cleanup_called_when_loras_not_loaded(
		self, mock_service: MockServiceFixture, sample_config: GeneratorConfig, mock_db: Mock
	) -> None:
		"""Test cleanup with loras_loaded=False."""
		service, _, mock_resource_manager, mock_lora_loader, *_ = mock_service
		mock_lora_loader.load_loras_for_generation.return_value = False

		service.generator.execute_pipeline = AsyncMock(
			return_value=Mock(images=[Image.new('RGB', (64, 64))], nsfw_content_detected=[False])
		)

		await service.generate_image(sample_config, mock_db)

		call_args = mock_resource_manager.cleanup_after_generation.call_args
		assert call_args.kwargs['loras_loaded'] is False
