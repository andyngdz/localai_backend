from __future__ import annotations

import importlib
import logging
import sys
from types import ModuleType
from unittest.mock import ANY, MagicMock, patch

import pytest


@pytest.fixture
def mock_dependencies():
	# Install lightweight stubs for heavy third-party modules before importing target module
	# Stub torch to avoid importing real PyTorch
	torch_stub = ModuleType('torch')
	# dtypes
	torch_stub.float16 = 'float16'
	torch_stub.float32 = 'float32'

	# cuda API
	class _Cuda:
		@staticmethod
		def is_available():
			return False

		@staticmethod
		def get_device_name(index: int = 0):
			return 'Fake CUDA Device'

		@staticmethod
		def device_count():
			return 0

		@staticmethod
		def current_device():
			return 0

		@staticmethod
		def get_device_properties(index: int = 0):
			return None

	torch_stub.cuda = _Cuda()

	# backends.mps
	class _Mps:
		@staticmethod
		def is_available():
			return False

	class _Backends:
		mps = _Mps()

	torch_stub.backends = _Backends()

	# minimal _C namespace used in some internal checks
	class _C:
		@staticmethod
		def _is_torch_function_enabled():
			return False

		_disabled_torch_function_impl = object()

	torch_stub._C = _C()
	sys.modules.setdefault('torch', torch_stub)

	# Stub transformers
	transformers_stub = ModuleType('transformers')

	class _CLIPImageProcessor:  # minimal placeholder
		@classmethod
		def from_pretrained(cls, *args, **kwargs):
			return MagicMock(name='CLIPImageProcessorInstance')

	transformers_stub.CLIPImageProcessor = _CLIPImageProcessor

	class _CLIPTokenizer:  # minimal placeholder
		@classmethod
		def from_pretrained(cls, *args, **kwargs):
			return MagicMock(name='CLIPTokenizerInstance')

	transformers_stub.CLIPTokenizer = _CLIPTokenizer
	sys.modules.setdefault('transformers', transformers_stub)

	# Stub diffusers and submodules
	diffusers_stub = ModuleType('diffusers')
	pipelines_stub = ModuleType('diffusers.pipelines')
	auto_pipeline_stub = ModuleType('diffusers.pipelines.auto_pipeline')
	sd_stub = ModuleType('diffusers.pipelines.stable_diffusion')
	sd_checker_stub = ModuleType('diffusers.pipelines.stable_diffusion.safety_checker')

	class _AutoPipelineForText2Image:  # minimal placeholder
		@classmethod
		def from_pretrained(cls, *args, **kwargs):  # will be patched
			return MagicMock(name='AutoPipelinePipe')

		@classmethod
		def from_single_file(cls, *args, **kwargs):  # will be patched
			return MagicMock(name='AutoPipelinePipeFromSingleFile')

	class _StableDiffusionSafetyChecker:  # minimal placeholder
		@classmethod
		def from_pretrained(cls, *args, **kwargs):
			return MagicMock(name='StableDiffusionSafetyCheckerInstance')

	auto_pipeline_stub.AutoPipelineForText2Image = _AutoPipelineForText2Image
	sd_checker_stub.StableDiffusionSafetyChecker = _StableDiffusionSafetyChecker

	# Populate scheduler classes on the diffusers root stub to satisfy
	# `from diffusers import <Schedulers...>` in app.cores.samplers.schedulers
	scheduler_names = [
		'DDIMScheduler',
		'DDPMScheduler',
		'DEISMultistepScheduler',
		'DPMSolverMultistepScheduler',
		'DPMSolverSDEScheduler',
		'DPMSolverSinglestepScheduler',
		'EulerAncestralDiscreteScheduler',
		'EulerDiscreteScheduler',
		'KDPM2AncestralDiscreteScheduler',
		'KDPM2DiscreteScheduler',
		'LMSDiscreteScheduler',
		'PNDMScheduler',
		'UniPCMultistepScheduler',
	]
	for name in scheduler_names:
		setattr(diffusers_stub, name, type(name, (), {}))

	sys.modules.setdefault('diffusers', diffusers_stub)
	sys.modules.setdefault('diffusers.pipelines', pipelines_stub)
	sys.modules.setdefault('diffusers.pipelines.auto_pipeline', auto_pipeline_stub)
	sys.modules.setdefault('diffusers.pipelines.stable_diffusion', sd_stub)
	sys.modules.setdefault('diffusers.pipelines.stable_diffusion.safety_checker', sd_checker_stub)

	# Import target module only after stubs are in place
	target_module = importlib.import_module('app.cores.model_loader.model_loader')

	with (
		patch.object(target_module, 'SessionLocal') as mock_session,
		patch.object(target_module, 'MaxMemoryConfig') as mock_max_memory,
		patch.object(target_module, 'device_service') as mock_device_service,
		patch.object(target_module, 'storage_service') as mock_storage_service,
		patch.object(target_module, 'CLIPImageProcessor') as mock_clip_processor,
		patch.object(target_module, 'StableDiffusionSafetyChecker') as mock_safety_checker,
		patch.object(target_module, 'AutoPipelineForText2Image') as mock_auto_pipeline,
		patch.object(target_module, 'socket_service') as mock_socket_service,
		patch.object(target_module, 'logger') as mock_logger,
	):
		# Arrange
		mock_device_service.device = 'cpu'
		mock_device_service.torch_dtype = 'torch.float16'
		mock_device_service.is_cuda = False
		mock_device_service.is_mps = False

		mock_max_memory.return_value.to_dict.return_value = {'cpu_offload': True}

		mock_storage_service.get_model_dir.return_value = '/fake/cache/path'

		mock_pipe = MagicMock()
		mock_auto_pipeline.from_pretrained.return_value = mock_pipe
		mock_auto_pipeline.from_single_file.return_value = mock_pipe
		# Set up to_empty to return the same mock_pipe
		mock_pipe.to_empty.return_value = mock_pipe
		# Set up to as fallback
		mock_pipe.to.return_value = mock_pipe

		yield {
			'mock_session': mock_session,
			'mock_max_memory': mock_max_memory,
			'mock_device_service': mock_device_service,
			'mock_storage_service': mock_storage_service,
			'mock_clip_processor': mock_clip_processor,
			'mock_safety_checker': mock_safety_checker,
			'mock_auto_pipeline': mock_auto_pipeline,
			'mock_socket_service': mock_socket_service,
			'mock_logger': mock_logger,
			'mock_pipe': mock_pipe,
			'module': target_module,
			'model_loader': getattr(target_module, 'model_loader'),
			'move_to_device': getattr(target_module, 'move_to_device'),
			'find_single_file_checkpoint': getattr(target_module, 'find_single_file_checkpoint'),
		}


def test_model_loader_success(mock_dependencies):
	# Arrange
	model_id = 'test-model-id'
	mock_pipe = mock_dependencies['mock_pipe']
	mock_auto_pipeline = mock_dependencies['mock_auto_pipeline']
	mock_socket_service = mock_dependencies['mock_socket_service']
	model_loader = mock_dependencies['model_loader']

	# Act
	result = model_loader(model_id)

	# Assert
	assert result == mock_pipe
	mock_auto_pipeline.from_pretrained.assert_called_once()
	call_args = mock_auto_pipeline.from_pretrained.call_args[0]
	assert call_args[0] == model_id
	mock_socket_service.model_load_completed.assert_called_once()


def test_model_loader_exception(mock_dependencies):
	# Arrange
	model_id = 'test-model-id'
	mock_auto_pipeline = mock_dependencies['mock_auto_pipeline']
	mock_socket_service = mock_dependencies['mock_socket_service']
	model_loader = mock_dependencies['model_loader']
	test_exception = RuntimeError('Model loading failed')
	mock_auto_pipeline.from_pretrained.side_effect = test_exception

	# Act & Assert
	with pytest.raises(RuntimeError, match='Model loading failed'):
		model_loader(model_id)

	# Assert that model_load_failed was called
	mock_socket_service.model_load_failed.assert_called_once_with(ANY)
	call_args, _ = mock_socket_service.model_load_failed.call_args
	assert call_args[0].id == model_id
	assert call_args[0].error == str(test_exception)


def test_move_to_device_to_empty_success(mock_dependencies):
	# Arrange
	move_to_device = mock_dependencies['move_to_device']
	mock_pipe = MagicMock()
	device = 'cpu'
	log_prefix = 'Test model'

	# Act
	result = move_to_device(mock_pipe, device, log_prefix)

	# Assert
	assert result == mock_pipe.to_empty.return_value
	mock_pipe.to_empty.assert_called_once_with(device)
	mock_pipe.to.assert_not_called()


def test_move_to_device_fallback_to_to(mock_dependencies):
	# Arrange
	move_to_device = mock_dependencies['move_to_device']
	mock_pipe = MagicMock()
	device = 'cpu'
	log_prefix = 'Test model'
	mock_pipe.to_empty.side_effect = AttributeError('to_empty not available')

	# Act
	result = move_to_device(mock_pipe, device, log_prefix)

	# Assert
	assert result == mock_pipe.to.return_value
	mock_pipe.to_empty.assert_called_once_with(device)
	mock_pipe.to.assert_called_once_with(device)


def test_move_to_device_fallback_to_to_type_error(mock_dependencies):
	# Arrange
	move_to_device = mock_dependencies['move_to_device']
	mock_pipe = MagicMock()
	device = 'cpu'
	log_prefix = 'Test model'
	mock_pipe.to_empty.side_effect = TypeError('Incompatible type')

	# Act
	result = move_to_device(mock_pipe, device, log_prefix)

	# Assert
	assert result == mock_pipe.to.return_value
	mock_pipe.to_empty.assert_called_once_with(device)
	mock_pipe.to.assert_called_once_with(device)


def test_move_to_device_logs_to_empty_success(mock_dependencies, caplog: pytest.LogCaptureFixture):
	# Arrange
	move_to_device = mock_dependencies['move_to_device']
	mock_pipe = MagicMock()
	device = 'cpu'
	log_prefix = 'Test model'
	mock_logger = mock_dependencies['mock_logger']

	# Act
	_ = move_to_device(mock_pipe, device, log_prefix)

	# Assert
	messages = [str(call.args[0]) for call in mock_logger.info.call_args_list]
	assert any('moved to cpu device using to_empty()' in m for m in messages)


def test_move_to_device_logs_fallback_attribute_error(mock_dependencies, caplog: pytest.LogCaptureFixture):
	# Arrange
	move_to_device = mock_dependencies['move_to_device']
	mock_pipe = MagicMock()
	device = 'cpu'
	log_prefix = 'Test model'
	mock_pipe.to_empty.side_effect = AttributeError('to_empty not available')
	mock_logger = mock_dependencies['mock_logger']

	# Act
	_ = move_to_device(mock_pipe, device, log_prefix)

	# Assert
	messages = [str(call.args[0]) for call in mock_logger.info.call_args_list]
	assert any('moved to cpu device using to()' in m for m in messages)


def test_model_loader_environment_error_fallback_success(mock_dependencies, caplog: pytest.LogCaptureFixture):
	# Arrange
	model_id = 'test-model-id'
	mock_auto_pipeline = mock_dependencies['mock_auto_pipeline']
	mock_socket_service = mock_dependencies['mock_socket_service']
	mock_pipe = mock_dependencies['mock_pipe']
	model_loader = mock_dependencies['model_loader']

	# First call raises EnvironmentError, second returns pipe (fallback to strategy 2)
	call_sequence: list[MagicMock | Exception] = [EnvironmentError('variant not available'), mock_pipe]

	def side_effect(*args, **kwargs):
		result = call_sequence.pop(0)
		if isinstance(result, Exception):
			raise result
		return result

	mock_auto_pipeline.from_pretrained.side_effect = side_effect

	# Act
	result = model_loader(model_id)

	# Assert
	assert result is mock_pipe
	assert mock_auto_pipeline.from_pretrained.call_count == 2
	# Verify the second call used use_safetensors=True (strategy 2 after strategy 1 fails)
	_, second_kwargs = mock_auto_pipeline.from_pretrained.call_args
	assert second_kwargs.get('use_safetensors') is True
	# Strategy 2 should not have 'variant' parameter
	assert 'variant' not in second_kwargs
	mock_socket_service.model_load_completed.assert_called_once()


def test_model_loader_environment_error_then_failure(mock_dependencies):
	# Arrange
	model_id = 'test-model-id'
	mock_auto_pipeline = mock_dependencies['mock_auto_pipeline']
	mock_socket_service = mock_dependencies['mock_socket_service']
	model_loader = mock_dependencies['model_loader']

	# All 4 strategies fail, last one raises RuntimeError
	call_sequence = [
		EnvironmentError('strategy 1 failed'),
		EnvironmentError('strategy 2 failed'),
		EnvironmentError('strategy 3 failed'),
		RuntimeError('all strategies failed'),
	]

	def side_effect(*args, **kwargs):
		result = call_sequence.pop(0)
		raise result

	mock_auto_pipeline.from_pretrained.side_effect = side_effect

	# Act & Assert
	with pytest.raises(RuntimeError, match='all strategies failed'):
		model_loader(model_id)

	# Ensure all 4 strategies were attempted
	assert mock_auto_pipeline.from_pretrained.call_count == 4
	# Ensure failure was emitted with correct payload
	mock_socket_service.model_load_failed.assert_called_once()
	args, _ = mock_socket_service.model_load_failed.call_args
	assert args[0].id == model_id
	assert args[0].error == 'all strategies failed'


def test_model_loader_calls_reset_device_map_when_available(mock_dependencies, caplog: pytest.LogCaptureFixture):
	# Arrange
	model_id = 'test-model-id'
	mock_pipe = mock_dependencies['mock_pipe']
	mock_pipe.reset_device_map = MagicMock()
	model_loader = mock_dependencies['model_loader']

	# Act
	_ = model_loader(model_id)

	# Assert
	mock_pipe.reset_device_map.assert_called_once_with()
	messages = [str(call.args[0]) for call in mock_dependencies['mock_logger'].info.call_args_list]
	assert any('Reset device map for pipeline' in m for m in messages)


def test_model_loader_applies_cuda_optimizations(mock_dependencies, caplog: pytest.LogCaptureFixture):
	# Arrange
	model_id = 'test-model-id'
	mock_pipe = mock_dependencies['mock_pipe']
	mock_device_service = mock_dependencies['mock_device_service']
	mock_device_service.is_cuda = True
	mock_device_service.is_mps = False
	model_loader = mock_dependencies['model_loader']
	# use mock logger

	# Act
	_ = model_loader(model_id)

	# Assert
	mock_pipe.enable_attention_slicing.assert_called_once_with()
	messages = [str(call.args[0]) for call in mock_dependencies['mock_logger'].info.call_args_list]
	assert any('Applied CUDA optimizations' in m for m in messages)


def test_model_loader_applies_mps_optimizations(mock_dependencies, caplog: pytest.LogCaptureFixture):
	# Arrange
	model_id = 'test-model-id'
	mock_pipe = mock_dependencies['mock_pipe']
	mock_device_service = mock_dependencies['mock_device_service']
	mock_device_service.is_cuda = False
	mock_device_service.is_mps = True
	model_loader = mock_dependencies['model_loader']
	# use mock logger

	# Act
	_ = model_loader(model_id)

	# Assert
	mock_pipe.enable_attention_slicing.assert_called_once_with()
	messages = [str(call.args[0]) for call in mock_dependencies['mock_logger'].info.call_args_list]
	assert any('Applied MPS optimizations' in m for m in messages)


def test_model_loader_applies_cpu_optimizations_by_default(mock_dependencies, caplog: pytest.LogCaptureFixture):
	# Arrange
	model_id = 'test-model-id'
	mock_pipe = mock_dependencies['mock_pipe']
	mock_device_service = mock_dependencies['mock_device_service']
	mock_device_service.is_cuda = False
	mock_device_service.is_mps = False
	model_loader = mock_dependencies['model_loader']
	caplog.set_level(logging.INFO, logger='app.cores.model_loader.model_loader')

	# Act
	_ = model_loader(model_id)

	# Assert
	mock_pipe.enable_attention_slicing.assert_called_once_with()
	messages = [str(call.args[0]) for call in mock_dependencies['mock_logger'].info.call_args_list]
	assert any('Applied CPU optimizations' in m for m in messages)


def test_move_to_device_logs_fallback_type_error(mock_dependencies, caplog: pytest.LogCaptureFixture):
	# Arrange
	move_to_device = mock_dependencies['move_to_device']
	mock_pipe = MagicMock()
	device = 'cpu'
	log_prefix = 'Test model'
	mock_pipe.to_empty.side_effect = TypeError('Incompatible type')
	mock_logger = mock_dependencies['mock_logger']

	# Act
	_ = move_to_device(mock_pipe, device, log_prefix)

	# Assert
	messages = [str(call.args[0]) for call in mock_logger.info.call_args_list]
	assert any('moved to cpu device using to()' in m for m in messages)


class TestFindSingleFileCheckpoint:
	"""Test the find_single_file_checkpoint function for detecting single-file checkpoints."""

	def test_returns_checkpoint_path_when_safetensors_exists(self, mock_dependencies, tmp_path):
		# Arrange
		find_single_file_checkpoint = mock_dependencies['find_single_file_checkpoint']
		model_dir = tmp_path / 'model'
		model_dir.mkdir()
		checkpoint_file = model_dir / 'model.safetensors'
		checkpoint_file.touch()

		# Act
		result = find_single_file_checkpoint(str(model_dir))

		# Assert
		assert result is not None
		assert result == str(checkpoint_file)
		assert result.endswith('.safetensors')

	def test_returns_none_when_no_checkpoint_files_exist(self, mock_dependencies, tmp_path):
		# Arrange
		find_single_file_checkpoint = mock_dependencies['find_single_file_checkpoint']
		model_dir = tmp_path / 'model'
		model_dir.mkdir()
		# Create a non-checkpoint file
		(model_dir / 'config.json').touch()

		# Act
		result = find_single_file_checkpoint(str(model_dir))

		# Assert
		assert result is None

	def test_returns_none_when_model_path_does_not_exist(self, mock_dependencies):
		# Arrange
		find_single_file_checkpoint = mock_dependencies['find_single_file_checkpoint']
		non_existent_path = '/non/existent/path'

		# Act
		result = find_single_file_checkpoint(non_existent_path)

		# Assert
		assert result is None

	def test_returns_first_checkpoint_when_multiple_exist(self, mock_dependencies, tmp_path):
		# Arrange
		find_single_file_checkpoint = mock_dependencies['find_single_file_checkpoint']
		model_dir = tmp_path / 'model'
		model_dir.mkdir()
		checkpoint1 = model_dir / 'model_v1.safetensors'
		checkpoint2 = model_dir / 'model_v2.safetensors'
		checkpoint1.touch()
		checkpoint2.touch()

		# Act
		result = find_single_file_checkpoint(str(model_dir))

		# Assert
		assert result is not None
		assert result.endswith('.safetensors')
		# Should return one of the checkpoint files
		assert result in [str(checkpoint1), str(checkpoint2)]


class TestModelLoadingStrategies:
	"""Test model loading strategies including single-file checkpoint support and enum usage."""

	def test_uses_single_file_strategy_when_checkpoint_found(self, mock_dependencies, tmp_path):
		# Arrange
		model_loader = mock_dependencies['model_loader']
		mock_auto_pipeline = mock_dependencies['mock_auto_pipeline']
		mock_storage_service = mock_dependencies['mock_storage_service']
		mock_pipe = mock_dependencies['mock_pipe']

		# Create a fake checkpoint in snapshots directory
		model_dir = tmp_path / 'models--org--model'
		snapshots_dir = model_dir / 'snapshots' / 'abc123'
		snapshots_dir.mkdir(parents=True)
		checkpoint_file = snapshots_dir / 'model.safetensors'
		checkpoint_file.touch()

		mock_storage_service.get_model_dir.return_value = str(model_dir)

		# Act
		result = model_loader('test-model')

		# Assert
		assert result == mock_pipe
		# from_single_file should be called since checkpoint was found
		mock_auto_pipeline.from_single_file.assert_called_once()
		call_args, call_kwargs = mock_auto_pipeline.from_single_file.call_args
		assert call_args[0] == str(checkpoint_file)

	def test_uses_pretrained_strategies_when_no_checkpoint_found(self, mock_dependencies):
		# Arrange
		model_loader = mock_dependencies['model_loader']
		mock_auto_pipeline = mock_dependencies['mock_auto_pipeline']
		mock_storage_service = mock_dependencies['mock_storage_service']
		mock_pipe = mock_dependencies['mock_pipe']

		# No checkpoint exists
		mock_storage_service.get_model_dir.return_value = '/non/existent/path'

		# Act
		result = model_loader('test-model')

		# Assert
		assert result == mock_pipe
		# from_single_file should NOT be called
		mock_auto_pipeline.from_single_file.assert_not_called()
		# from_pretrained should be called
		mock_auto_pipeline.from_pretrained.assert_called()

	def test_uses_model_loading_strategy_enum(self, mock_dependencies, tmp_path):
		# Arrange
		from app.cores.constants.model_loader import ModelLoadingStrategy

		model_loader = mock_dependencies['model_loader']
		mock_auto_pipeline = mock_dependencies['mock_auto_pipeline']
		mock_storage_service = mock_dependencies['mock_storage_service']
		mock_logger = mock_dependencies['mock_logger']

		# Create a fake checkpoint
		model_dir = tmp_path / 'models--org--model'
		snapshots_dir = model_dir / 'snapshots' / 'abc123'
		snapshots_dir.mkdir(parents=True)
		checkpoint_file = snapshots_dir / 'model.safetensors'
		checkpoint_file.touch()

		mock_storage_service.get_model_dir.return_value = str(model_dir)

		# Act
		_ = model_loader('test-model')

		# Assert - check logger was called with enum value
		log_messages = [str(call.args[0]) for call in mock_logger.info.call_args_list]
		# Should log the single_file strategy type
		assert any(ModelLoadingStrategy.SINGLE_FILE in msg for msg in log_messages)

	def test_fallback_to_pretrained_when_single_file_fails(self, mock_dependencies, tmp_path):
		# Arrange
		model_loader = mock_dependencies['model_loader']
		mock_auto_pipeline = mock_dependencies['mock_auto_pipeline']
		mock_storage_service = mock_dependencies['mock_storage_service']
		mock_pipe = mock_dependencies['mock_pipe']

		# Create a fake checkpoint
		model_dir = tmp_path / 'models--org--model'
		snapshots_dir = model_dir / 'snapshots' / 'abc123'
		snapshots_dir.mkdir(parents=True)
		checkpoint_file = snapshots_dir / 'model.safetensors'
		checkpoint_file.touch()

		mock_storage_service.get_model_dir.return_value = str(model_dir)

		# from_single_file fails, from_pretrained succeeds
		mock_auto_pipeline.from_single_file.side_effect = EnvironmentError('Single file failed')
		mock_auto_pipeline.from_pretrained.return_value = mock_pipe

		# Act
		result = model_loader('test-model')

		# Assert
		assert result == mock_pipe
		# Both methods should be called
		mock_auto_pipeline.from_single_file.assert_called_once()
		mock_auto_pipeline.from_pretrained.assert_called()

	def test_all_five_strategies_attempted_on_failure(self, mock_dependencies, tmp_path):
		# Arrange
		model_loader = mock_dependencies['model_loader']
		mock_auto_pipeline = mock_dependencies['mock_auto_pipeline']
		mock_storage_service = mock_dependencies['mock_storage_service']

		# Create a fake checkpoint to trigger 5 strategies (1 single-file + 4 pretrained)
		model_dir = tmp_path / 'models--org--model'
		snapshots_dir = model_dir / 'snapshots' / 'abc123'
		snapshots_dir.mkdir(parents=True)
		checkpoint_file = snapshots_dir / 'model.safetensors'
		checkpoint_file.touch()

		mock_storage_service.get_model_dir.return_value = str(model_dir)

		# All strategies fail
		mock_auto_pipeline.from_single_file.side_effect = EnvironmentError('Single file failed')
		mock_auto_pipeline.from_pretrained.side_effect = [
			EnvironmentError('Strategy 2 failed'),
			EnvironmentError('Strategy 3 failed'),
			EnvironmentError('Strategy 4 failed'),
			RuntimeError('All strategies failed'),
		]

		# Act & Assert
		with pytest.raises(RuntimeError, match='All strategies failed'):
			model_loader('test-model')

		# Should try single-file once + pretrained 4 times = 5 total attempts
		assert mock_auto_pipeline.from_single_file.call_count == 1
		assert mock_auto_pipeline.from_pretrained.call_count == 4


def test_model_loader_uses_storage_service_get_model_dir(mock_dependencies):
	"""Test that model_loader uses storage_service.get_model_dir() to get model cache path."""
	# Arrange
	model_id = 'test-model-id'
	model_loader = mock_dependencies['model_loader']
	mock_storage_service = mock_dependencies['mock_storage_service']
	mock_storage_service.get_model_dir.return_value = '/fake/cache/models--org--model'

	# Act
	_ = model_loader(model_id)

	# Assert
	mock_storage_service.get_model_dir.assert_called_once_with(model_id)


def test_model_loader_does_not_use_device_map_parameter(mock_dependencies):
	"""Test that model_loader does NOT use device_map parameter in loading calls."""
	# Arrange
	model_id = 'test-model-id'
	model_loader = mock_dependencies['model_loader']
	mock_auto_pipeline = mock_dependencies['mock_auto_pipeline']

	# Act
	_ = model_loader(model_id)

	# Assert - check all calls to from_pretrained
	for call_args in mock_auto_pipeline.from_pretrained.call_args_list:
		_, kwargs = call_args
		assert 'device_map' not in kwargs, 'device_map should not be in kwargs'


def test_model_loader_does_not_mutate_strategy_dictionaries(mock_dependencies):
	"""Test that model_loader does not mutate the original strategy dictionaries."""
	# Arrange
	model_id = 'test-model-id'
	model_loader = mock_dependencies['model_loader']
	mock_auto_pipeline = mock_dependencies['mock_auto_pipeline']

	# Track all kwargs passed to from_pretrained to ensure 'type' is never present
	kwargs_list = []

	def capture_kwargs(*args, **kwargs):
		kwargs_list.append(kwargs.copy())
		return mock_dependencies['mock_pipe']

	mock_auto_pipeline.from_pretrained.side_effect = capture_kwargs

	# Act
	_ = model_loader(model_id)

	# Assert - 'type' should never be in kwargs (it should be filtered out, not popped)
	for kwargs in kwargs_list:
		assert 'type' not in kwargs, "'type' should be filtered out, not present in kwargs"
