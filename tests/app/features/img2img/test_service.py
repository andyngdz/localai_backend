"""Tests for img2img service."""

from unittest.mock import Mock, patch

import pytest
import torch
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from PIL import Image

from app.cores.samplers import SamplerType
from app.features.img2img.schemas import ImageGenerationItem, ImageGenerationResponse, Img2ImgConfig


@pytest.fixture
def mock_img2img_service():
	"""Create Img2ImgService with mocked dependencies."""
	with (
		patch('app.features.img2img.service.model_manager') as mock_model_manager,
		patch('app.features.img2img.service.pipeline_converter') as mock_pipeline_converter,
		patch('app.features.img2img.service.seed_manager') as mock_seed_manager,
		patch('app.features.img2img.service.image_processor') as mock_image_processor,
		patch('app.features.img2img.service.memory_manager') as mock_memory_manager,
		patch('app.features.img2img.service.progress_callback') as mock_progress_callback,
		patch('app.features.img2img.service.image_service') as mock_image_service,
		patch('app.features.img2img.service.styles_service') as mock_styles_service,
		patch('app.features.img2img.service.torch') as mock_torch,
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

		from app.features.img2img.service import Img2ImgService

		service = Img2ImgService()

		yield (
			service,
			mock_model_manager,
			mock_pipeline_converter,
			mock_seed_manager,
			mock_image_processor,
			mock_memory_manager,
			mock_progress_callback,
			mock_image_service,
			mock_styles_service,
			mock_torch,
		)


@pytest.fixture
def sample_img2img_config():
	"""Create sample Img2ImgConfig for testing."""
	return Img2ImgConfig(
		init_image='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
		strength=0.75,
		resize_mode='resize',
		prompt='test prompt',
		width=512,
		height=512,
		steps=20,
		cfg_scale=7.5,
		number_of_images=1,
		seed=-1,
		sampler=SamplerType.EULER_A,
		styles=[],
	)


class TestImg2ImgServiceInit:
	def test_creates_executor(self, mock_img2img_service):
		service, *_ = mock_img2img_service
		assert service.executor is not None
		assert hasattr(service.executor, 'submit')


class TestGenerateImageFromImage:
	@pytest.mark.asyncio
	async def test_raises_error_when_no_model_loaded(self, mock_img2img_service, sample_img2img_config):
		service, mock_model_manager, *_ = mock_img2img_service
		mock_model_manager.pipe = None

		with pytest.raises(ValueError, match='No model is currently loaded'):
			await service.generate_image_from_image(sample_img2img_config)

	@pytest.mark.asyncio
	async def test_converts_pipeline_to_img2img(self, mock_img2img_service, sample_img2img_config):
		service, mock_model_manager, mock_pipeline_converter, *_ = mock_img2img_service
		mock_model_manager.pipe = Mock()
		mock_converted_pipe = Mock()
		mock_pipeline_converter.convert_to_img2img.return_value = mock_converted_pipe

		# Mock to stop execution after conversion
		service.executor = Mock()
		service.executor.submit = Mock(side_effect=Exception('Stop'))

		try:
			await service.generate_image_from_image(sample_img2img_config)
		except Exception:
			pass

		mock_pipeline_converter.convert_to_img2img.assert_called_once()
		assert mock_model_manager.pipe == mock_converted_pipe

	@pytest.mark.asyncio
	async def test_clears_cache_before_generation(self, mock_img2img_service, sample_img2img_config):
		service, mock_model_manager, _, _, _, mock_memory_manager, *_ = mock_img2img_service
		mock_model_manager.pipe = Mock()

		service.executor = Mock()
		service.executor.submit = Mock(side_effect=Exception('Stop'))

		try:
			await service.generate_image_from_image(sample_img2img_config)
		except Exception:
			pass

		mock_memory_manager.clear_cache.assert_called()

	@pytest.mark.asyncio
	async def test_validates_batch_size(self, mock_img2img_service, sample_img2img_config):
		service, mock_model_manager, _, _, _, mock_memory_manager, *_ = mock_img2img_service
		mock_model_manager.pipe = Mock()

		service.executor = Mock()
		service.executor.submit = Mock(side_effect=Exception('Stop'))

		try:
			await service.generate_image_from_image(sample_img2img_config)
		except Exception:
			pass

		mock_memory_manager.validate_batch_size.assert_called_once_with(1, 512, 512)

	@pytest.mark.asyncio
	async def test_decodes_base64_image(self, mock_img2img_service, sample_img2img_config):
		service, mock_model_manager, _, _, _, _, _, mock_image_service, mock_styles_service, _ = mock_img2img_service
		mock_model_manager.pipe = Mock()
		test_image = Image.new('RGB', (64, 64), color='blue')
		mock_image_service.from_base64.return_value = test_image
		mock_image_service.resize_image.return_value = test_image

		service.executor = Mock()
		service.executor.submit = Mock(side_effect=Exception('Stop'))

		try:
			await service.generate_image_from_image(sample_img2img_config)
		except Exception:
			pass

		mock_image_service.from_base64.assert_called_once()
		mock_image_service.resize_image.assert_called_once()

	@pytest.mark.asyncio
	async def test_applies_styles(self, mock_img2img_service, sample_img2img_config):
		service, mock_model_manager, _, _, _, _, _, _, mock_styles_service, _ = mock_img2img_service
		mock_model_manager.pipe = Mock()
		mock_styles_service.apply_styles.return_value = ('positive', 'negative')

		service.executor = Mock()
		service.executor.submit = Mock(side_effect=Exception('Stop'))

		try:
			await service.generate_image_from_image(sample_img2img_config)
		except Exception:
			pass

		mock_styles_service.apply_styles.assert_called_once()

	@pytest.mark.asyncio
	async def test_successful_generation(self, mock_img2img_service, sample_img2img_config):
		(
			service,
			mock_model_manager,
			mock_pipeline_converter,
			_,
			mock_image_processor,
			_,
			_,
			mock_image_service,
			mock_styles_service,
			_,
		) = mock_img2img_service

		# Mock the pipe
		mock_pipe = Mock()
		mock_pipe.device = 'cpu'
		test_image = Image.new('RGB', (64, 64), color='blue')
		mock_pipe.return_value = StableDiffusionPipelineOutput(images=[test_image], nsfw_content_detected=[False])
		mock_model_manager.pipe = mock_pipe
		mock_pipeline_converter.convert_to_img2img.return_value = mock_pipe

		# Mock image service
		mock_image_service.from_base64.return_value = test_image
		mock_image_service.resize_image.return_value = test_image

		# Mock styles
		mock_styles_service.apply_styles.return_value = ('positive', 'negative')

		result = await service.generate_image_from_image(sample_img2img_config)

		assert isinstance(result, ImageGenerationResponse)
		assert len(result.items) == 1
		assert isinstance(result.items[0], ImageGenerationItem)
		assert result.items[0].path == '/static/test.png'
		assert result.nsfw_content_detected == [False]

	@pytest.mark.asyncio
	async def test_handles_oom_error(self, mock_img2img_service, sample_img2img_config):
		(
			service,
			mock_model_manager,
			mock_pipeline_converter,
			_,
			_,
			mock_memory_manager,
			_,
			mock_image_service,
			mock_styles_service,
			_,
		) = mock_img2img_service

		mock_pipe = Mock()
		mock_pipe.device = 'cuda'
		mock_pipe.side_effect = torch.cuda.OutOfMemoryError('CUDA out of memory')
		mock_model_manager.pipe = mock_pipe
		mock_pipeline_converter.convert_to_img2img.return_value = mock_pipe

		test_image = Image.new('RGB', (64, 64), color='blue')
		mock_image_service.from_base64.return_value = test_image
		mock_image_service.resize_image.return_value = test_image
		mock_styles_service.apply_styles.return_value = ('pos', 'neg')

		with pytest.raises(ValueError, match='Out of memory'):
			await service.generate_image_from_image(sample_img2img_config)

		# Verify cache was cleared
		assert mock_memory_manager.clear_cache.call_count >= 2

	@pytest.mark.asyncio
	async def test_handles_invalid_base64_image(self, mock_img2img_service, sample_img2img_config):
		service, mock_model_manager, _, _, _, _, _, mock_image_service, *_ = mock_img2img_service
		mock_model_manager.pipe = Mock()
		mock_image_service.from_base64.side_effect = ValueError('Invalid base64')

		with pytest.raises(ValueError, match='Invalid base64'):
			await service.generate_image_from_image(sample_img2img_config)

	@pytest.mark.asyncio
	async def test_clears_cache_in_finally_block(self, mock_img2img_service, sample_img2img_config):
		service, mock_model_manager, _, _, _, mock_memory_manager, _, mock_image_service, mock_styles_service, _ = (
			mock_img2img_service
		)

		mock_pipe = Mock()
		mock_pipe.device = 'cpu'
		mock_pipe.side_effect = RuntimeError('Test error')
		mock_model_manager.pipe = mock_pipe

		test_image = Image.new('RGB', (64, 64), color='blue')
		mock_image_service.from_base64.return_value = test_image
		mock_image_service.resize_image.return_value = test_image
		mock_styles_service.apply_styles.return_value = ('pos', 'neg')

		try:
			await service.generate_image_from_image(sample_img2img_config)
		except Exception:
			pass

		# Verify cache was cleared in finally block
		mock_memory_manager.clear_cache.assert_called()
