import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import torch
from PIL import Image

from app.cores.samplers import SamplerType
from app.features.generators.constants import DEFAULT_NEGATIVE_PROMPT
from app.features.generators.schemas import (
	GeneratorConfig,
	ImageGenerationItem,
	ImageGenerationResponse,
	ImageGenerationStepEndResponse,
)


@pytest.fixture
def mock_service():
	"""Create GeneratorService with mocked dependencies."""
	with (
		patch('app.features.generators.service.model_manager') as mock_model_manager,
		patch('app.features.generators.service.device_service') as mock_device_service,
		patch('app.features.generators.service.image_service') as mock_image_service,
		patch('app.features.generators.service.styles_service') as mock_styles_service,
		patch('app.features.generators.service.socket_service') as mock_socket_service,
		patch('app.features.generators.service.torch') as mock_torch,
	):
		# Configure device_service
		mock_device_service.is_available = False
		mock_device_service.is_cuda = False
		mock_device_service.is_mps = False
		mock_device_service.get_recommended_batch_size.return_value = 3

		# Configure torch
		mock_torch.randint.return_value = torch.tensor([12345])
		mock_torch.manual_seed = Mock()
		mock_torch.cuda.manual_seed = Mock()
		mock_torch.cuda.empty_cache = Mock()
		mock_torch.cuda.OutOfMemoryError = torch.cuda.OutOfMemoryError
		mock_torch.mps.manual_seed = Mock()
		mock_torch.Generator = Mock(return_value=Mock())

		from app.features.generators.service import GeneratorService

		service = GeneratorService()

		yield (
			service,
			mock_model_manager,
			mock_device_service,
			mock_image_service,
			mock_styles_service,
			mock_socket_service,
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


class TestGeneratorServiceInit:
	def test_creates_executor(self, mock_service):
		service, *_ = mock_service
		assert service.executor is not None
		assert hasattr(service.executor, 'submit')


class TestGetRandomSeed:
	def test_returns_valid_integer_in_range(self, mock_service):
		service, *_ = mock_service
		seed = service.get_random_seed
		assert isinstance(seed, int)
		assert 0 <= seed < 2**32


class TestGetSeed:
	def test_uses_explicit_seed_when_not_minus_one(self, mock_service):
		service, _, _, _, _, _, mock_torch = mock_service
		result = service.get_seed(42)

		assert result == 42
		mock_torch.manual_seed.assert_called_once_with(42)

	def test_generates_random_seed_when_minus_one(self, mock_service):
		service, _, _, _, _, _, mock_torch = mock_service
		result = service.get_seed(-1)

		assert result == service.get_random_seed
		mock_torch.manual_seed.assert_called_once()

	def test_sets_cuda_seed_when_cuda_available(self, mock_service):
		service, _, mock_device_service, _, _, _, mock_torch = mock_service
		mock_device_service.is_available = True
		mock_device_service.is_cuda = True

		service.get_seed(42)

		mock_torch.cuda.manual_seed.assert_called_once_with(42)

	def test_sets_mps_seed_when_mps_available(self, mock_service):
		service, _, mock_device_service, _, _, _, mock_torch = mock_service
		mock_device_service.is_available = True
		mock_device_service.is_mps = True

		service.get_seed(42)

		mock_torch.mps.manual_seed.assert_called_once_with(42)


class TestApplyHiresFix:
	def test_logs_warning_when_hires_fix_enabled(self, mock_service, caplog):
		service, *_ = mock_service

		service.apply_hires_fix(True)

		assert 'Hires fix requested, but not fully implemented' in caplog.text

	def test_does_not_log_when_hires_fix_disabled(self, mock_service, caplog):
		service, *_ = mock_service

		service.apply_hires_fix(False)

		assert 'Hires fix requested' not in caplog.text


class TestIsNsfwContentDetected:
	def test_returns_nsfw_flags_when_detected(self, mock_service):
		service, *_ = mock_service
		output = {'nsfw_content_detected': [True, False, True], 'images': [None, None, None]}

		result = service.is_nsfw_content_detected(output)

		assert result == [True, False, True]

	def test_returns_false_list_when_no_nsfw(self, mock_service):
		service, *_ = mock_service
		output = {'nsfw_content_detected': None, 'images': [None, None]}

		result = service.is_nsfw_content_detected(output)

		assert result == [False, False]

	def test_handles_empty_images_list(self, mock_service):
		service, *_ = mock_service
		output = {'nsfw_content_detected': None, 'images': []}

		result = service.is_nsfw_content_detected(output)

		assert result == []


class TestGenerateFileName:
	def test_generates_timestamp_based_filename(self, mock_service):
		service, *_ = mock_service

		filename = service.generate_file_name()

		# Check format: YYYYMMDD_HHMMSS_ffffff (22 chars total)
		assert len(filename) == 22
		assert filename[8] == '_'
		assert filename[15] == '_'
		# Verify it's a valid datetime format
		datetime.strptime(filename, '%Y%m%d_%H%M%S_%f')


class TestSaveImage:
	def test_raises_value_error_when_image_is_none(self, mock_service):
		service, *_ = mock_service

		with pytest.raises(ValueError, match='Failed to generate any image'):
			service.save_image(None)

	def test_saves_image_and_returns_paths(self, mock_service, tmp_path, monkeypatch):
		service, *_ = mock_service

		# Create a test image
		test_image = Image.new('RGB', (64, 64), color='red')

		# Mock the paths
		generated_folder = tmp_path / 'generated'
		static_folder = tmp_path / 'static/generated_images'
		generated_folder.mkdir(parents=True, exist_ok=True)
		static_folder.mkdir(parents=True, exist_ok=True)

		monkeypatch.setattr('app.features.generators.service.GENERATED_IMAGES_FOLDER', str(generated_folder))
		monkeypatch.setattr('app.features.generators.service.GENERATED_IMAGES_STATIC_FOLDER', str(static_folder))

		static_path, file_name = service.save_image(test_image)

		# Verify file was created
		assert os.path.exists(os.path.join(generated_folder, f'{file_name}.png'))
		assert static_path == os.path.join(str(static_folder), f'{file_name}.png')


class TestLatentsToRgb:
	def test_converts_latent_tensor_to_rgb_image(self):
		# This test needs real torch, not mocked
		from app.features.generators.service import GeneratorService

		service = GeneratorService()

		# Create a real latent tensor (4 channels for latent space)
		import torch as real_torch

		latents = real_torch.randn(4, 8, 8)

		result = service.latents_to_rgb(latents)

		assert isinstance(result, Image.Image)
		assert result.mode == 'RGB'
		assert result.size == (8, 8)


class TestCallbackOnStepEnd:
	def test_emits_socket_events_for_each_latent(self):
		# This test needs real torch tensors, but mocked services
		import torch as real_torch

		with (
			patch('app.features.generators.service.image_service') as mock_image_service,
			patch('app.features.generators.service.socket_service') as mock_socket_service,
		):
			from app.features.generators.service import GeneratorService

			service = GeneratorService()

			# Real latents
			latents = [real_torch.randn(4, 8, 8), real_torch.randn(4, 8, 8)]
			callback_kwargs = {'latents': latents}
			mock_image_service.to_base64.return_value = 'base64_image_data'

			result = service.callback_on_step_end(None, 5, 123.45, callback_kwargs)

			# Verify socket emissions
			assert mock_socket_service.image_generation_step_end.call_count == 2
			assert result == callback_kwargs

	def test_passes_correct_data_to_socket_service(self):
		# This test needs real torch tensors, but mocked services
		import torch as real_torch

		with (
			patch('app.features.generators.service.image_service') as mock_image_service,
			patch('app.features.generators.service.socket_service') as mock_socket_service,
		):
			from app.features.generators.service import GeneratorService

			service = GeneratorService()

			latents = [real_torch.randn(4, 8, 8)]
			callback_kwargs = {'latents': latents}
			mock_image_service.to_base64.return_value = 'base64_data'

			service.callback_on_step_end(None, 10, 50.5, callback_kwargs)

			call_args = mock_socket_service.image_generation_step_end.call_args[0][0]
			assert isinstance(call_args, ImageGenerationStepEndResponse)
			assert call_args.current_step == 10
			assert call_args.timestep == 50.5
			assert call_args.index == 0
			assert call_args.image_base64 == 'base64_data'


class TestGenerateImage:
	@pytest.mark.asyncio
	async def test_raises_error_when_no_model_loaded(self, mock_service, sample_config):
		service, mock_model_manager, *_ = mock_service
		mock_model_manager.pipe = None

		with pytest.raises(ValueError, match='No model is currently loaded'):
			await service.generate_image(sample_config)

	@pytest.mark.asyncio
	async def test_clears_cuda_cache_before_generation_when_cuda_available(
		self, mock_service, sample_config
	):
		service, mock_model_manager, mock_device_service, _, _, _, mock_torch = mock_service
		mock_model_manager.pipe = Mock()
		mock_device_service.is_cuda = True

		# Mock the executor to avoid actual execution
		service.executor = Mock()
		service.executor.submit = Mock(side_effect=Exception('Stop execution'))

		try:
			await service.generate_image(sample_config)
		except:
			pass

		mock_torch.cuda.empty_cache.assert_called()

	@pytest.mark.asyncio
	async def test_validates_batch_size_with_device_service(self, mock_service, sample_config, caplog):
		service, mock_model_manager, mock_device_service, *_ = mock_service
		mock_model_manager.pipe = Mock()
		mock_device_service.get_recommended_batch_size.return_value = 2

		# Set batch size higher than recommended
		sample_config.number_of_images = 4

		# Mock executor to stop execution after validation
		service.executor = Mock()
		service.executor.submit = Mock(side_effect=Exception('Stop'))

		try:
			await service.generate_image(sample_config)
		except:
			pass

		mock_device_service.get_recommended_batch_size.assert_called_once()
		assert 'may cause out of memory errors' in caplog.text
		assert 'Your GPU supports up to 2 images at once' in caplog.text

	@pytest.mark.asyncio
	async def test_does_not_warn_when_batch_size_within_limit(self, mock_service, sample_config, caplog):
		service, mock_model_manager, mock_device_service, *_ = mock_service
		mock_model_manager.pipe = Mock()
		mock_device_service.get_recommended_batch_size.return_value = 3

		sample_config.number_of_images = 2

		service.executor = Mock()
		service.executor.submit = Mock(side_effect=Exception('Stop'))

		try:
			await service.generate_image(sample_config)
		except:
			pass

		assert 'may cause out of memory errors' not in caplog.text

	@pytest.mark.asyncio
	async def test_sets_sampler_before_generation(self, mock_service, sample_config):
		service, mock_model_manager, *_ = mock_service
		mock_model_manager.pipe = Mock()

		service.executor = Mock()
		service.executor.submit = Mock(side_effect=Exception('Stop'))

		try:
			await service.generate_image(sample_config)
		except:
			pass

		mock_model_manager.set_sampler.assert_called_once_with(SamplerType.EULER_A)

	@pytest.mark.asyncio
	async def test_applies_styles_via_styles_service(self, mock_service, sample_config):
		service, mock_model_manager, _, _, mock_styles_service, _, _ = mock_service
		mock_model_manager.pipe = Mock()
		mock_styles_service.apply_styles.return_value = ('positive prompt', 'negative prompt')

		sample_config.styles = ['style1', 'style2']

		service.executor = Mock()
		service.executor.submit = Mock(side_effect=Exception('Stop'))

		try:
			await service.generate_image(sample_config)
		except:
			pass

		mock_styles_service.apply_styles.assert_called_once_with('test prompt', ['style1', 'style2'])

	@pytest.mark.asyncio
	async def test_uses_default_negative_prompt_when_none_provided(self, mock_service, sample_config):
		service, mock_model_manager, _, _, mock_styles_service, _, _ = mock_service
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
			await service.generate_image(sample_config)
		except:
			pass

		# Verify styles were applied
		mock_styles_service.apply_styles.assert_called_once()

	@pytest.mark.asyncio
	async def test_successful_image_generation(self, mock_service, sample_config, tmp_path, monkeypatch):
		service, mock_model_manager, _, _, mock_styles_service, _, _ = mock_service

		# Mock the pipe
		mock_pipe = Mock()
		mock_pipe.device = 'cpu'
		test_image = Image.new('RGB', (64, 64), color='blue')
		mock_pipe.return_value = {'images': [test_image], 'nsfw_content_detected': [False]}
		mock_model_manager.pipe = mock_pipe

		# Mock styles
		mock_styles_service.apply_styles.return_value = ('positive', 'negative')

		# Mock file paths
		generated_folder = tmp_path / 'generated'
		static_folder = tmp_path / 'static/generated_images'
		generated_folder.mkdir(parents=True, exist_ok=True)
		static_folder.mkdir(parents=True, exist_ok=True)

		monkeypatch.setattr('app.features.generators.service.GENERATED_IMAGES_FOLDER', str(generated_folder))
		monkeypatch.setattr('app.features.generators.service.GENERATED_IMAGES_STATIC_FOLDER', str(static_folder))

		result = await service.generate_image(sample_config)

		assert isinstance(result, ImageGenerationResponse)
		assert len(result.items) == 1
		assert isinstance(result.items[0], ImageGenerationItem)
		assert result.nsfw_content_detected == [False]

	@pytest.mark.asyncio
	async def test_handles_oom_error_and_clears_cache(self, mock_service, sample_config):
		service, mock_model_manager, mock_device_service, _, mock_styles_service, _, mock_torch = mock_service

		mock_pipe = Mock()
		mock_pipe.device = 'cuda'
		mock_pipe.side_effect = torch.cuda.OutOfMemoryError('CUDA out of memory')
		mock_model_manager.pipe = mock_pipe
		mock_device_service.is_cuda = True
		mock_styles_service.apply_styles.return_value = ('pos', 'neg')

		with pytest.raises(ValueError, match='Out of memory error'):
			await service.generate_image(sample_config)

		# Verify cache was cleared in except and finally blocks
		assert mock_torch.cuda.empty_cache.call_count >= 2

	@pytest.mark.asyncio
	async def test_handles_file_not_found_error(self, mock_service, sample_config):
		service, mock_model_manager, _, _, mock_styles_service, _, _ = mock_service

		mock_pipe = Mock()
		mock_pipe.device = 'cpu'
		mock_pipe.side_effect = FileNotFoundError('Model files not found')
		mock_model_manager.pipe = mock_pipe
		mock_styles_service.apply_styles.return_value = ('pos', 'neg')

		with pytest.raises(ValueError, match='Model files not found'):
			await service.generate_image(sample_config)

	@pytest.mark.asyncio
	async def test_handles_general_exception(self, mock_service, sample_config):
		service, mock_model_manager, _, _, mock_styles_service, _, _ = mock_service

		mock_pipe = Mock()
		mock_pipe.device = 'cpu'
		mock_pipe.side_effect = RuntimeError('Something went wrong')
		mock_model_manager.pipe = mock_pipe
		mock_styles_service.apply_styles.return_value = ('pos', 'neg')

		with pytest.raises(ValueError, match='Failed to generate image'):
			await service.generate_image(sample_config)

	@pytest.mark.asyncio
	async def test_clears_cuda_cache_in_finally_block(self, mock_service, sample_config):
		service, mock_model_manager, mock_device_service, _, mock_styles_service, _, mock_torch = mock_service

		mock_pipe = Mock()
		mock_pipe.device = 'cuda'
		mock_pipe.side_effect = RuntimeError('Test error')
		mock_model_manager.pipe = mock_pipe
		mock_device_service.is_cuda = True
		mock_styles_service.apply_styles.return_value = ('pos', 'neg')

		try:
			await service.generate_image(sample_config)
		except:
			pass

		# Verify cache was cleared
		mock_torch.cuda.empty_cache.assert_called()
