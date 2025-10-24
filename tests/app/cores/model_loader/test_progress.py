from __future__ import annotations

import importlib
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

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

	class _DiffusionPipeline:  # minimal placeholder
		@classmethod
		def from_pretrained(cls, *args, **kwargs):  # will be patched
			return MagicMock(name='DiffusionPipe')

		@classmethod
		def from_single_file(cls, *args, **kwargs):  # will be patched
			return MagicMock(name='DiffusionPipeFromSingleFile')

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

	diffusers_stub.DiffusionPipeline = _DiffusionPipeline
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
			'map_step_to_phase': getattr(target_module, 'map_step_to_phase'),
			'emit_progress': getattr(target_module, 'emit_progress'),
		}


class TestMapStepToPhase:
	"""Test the map_step_to_phase function for correct phase mapping."""

	def test_step_1_maps_to_initialization(self, mock_dependencies):
		# Arrange
		from app.cores.model_loader.schemas import ModelLoadPhase

		map_step_to_phase = mock_dependencies['map_step_to_phase']

		# Act
		result = map_step_to_phase(1)

		# Assert
		assert result == ModelLoadPhase.INITIALIZATION

	def test_step_2_maps_to_initialization(self, mock_dependencies):
		# Arrange
		from app.cores.model_loader.schemas import ModelLoadPhase

		map_step_to_phase = mock_dependencies['map_step_to_phase']

		# Act
		result = map_step_to_phase(2)

		# Assert
		assert result == ModelLoadPhase.INITIALIZATION

	def test_step_3_maps_to_loading_model(self, mock_dependencies):
		# Arrange
		from app.cores.model_loader.schemas import ModelLoadPhase

		map_step_to_phase = mock_dependencies['map_step_to_phase']

		# Act
		result = map_step_to_phase(3)

		# Assert
		assert result == ModelLoadPhase.LOADING_MODEL

	def test_step_4_maps_to_loading_model(self, mock_dependencies):
		# Arrange
		from app.cores.model_loader.schemas import ModelLoadPhase

		map_step_to_phase = mock_dependencies['map_step_to_phase']

		# Act
		result = map_step_to_phase(4)

		# Assert
		assert result == ModelLoadPhase.LOADING_MODEL

	def test_step_5_maps_to_loading_model(self, mock_dependencies):
		# Arrange
		from app.cores.model_loader.schemas import ModelLoadPhase

		map_step_to_phase = mock_dependencies['map_step_to_phase']

		# Act
		result = map_step_to_phase(5)

		# Assert
		assert result == ModelLoadPhase.LOADING_MODEL

	def test_step_6_maps_to_device_setup(self, mock_dependencies):
		# Arrange
		from app.cores.model_loader.schemas import ModelLoadPhase

		map_step_to_phase = mock_dependencies['map_step_to_phase']

		# Act
		result = map_step_to_phase(6)

		# Assert
		assert result == ModelLoadPhase.DEVICE_SETUP

	def test_step_7_maps_to_device_setup(self, mock_dependencies):
		# Arrange
		from app.cores.model_loader.schemas import ModelLoadPhase

		map_step_to_phase = mock_dependencies['map_step_to_phase']

		# Act
		result = map_step_to_phase(7)

		# Assert
		assert result == ModelLoadPhase.DEVICE_SETUP

	def test_step_8_maps_to_optimization(self, mock_dependencies):
		# Arrange
		from app.cores.model_loader.schemas import ModelLoadPhase

		map_step_to_phase = mock_dependencies['map_step_to_phase']

		# Act
		result = map_step_to_phase(8)

		# Assert
		assert result == ModelLoadPhase.OPTIMIZATION

	def test_step_9_maps_to_optimization(self, mock_dependencies):
		# Arrange
		from app.cores.model_loader.schemas import ModelLoadPhase

		map_step_to_phase = mock_dependencies['map_step_to_phase']

		# Act
		result = map_step_to_phase(9)

		# Assert
		assert result == ModelLoadPhase.OPTIMIZATION


class TestModelLoadProgressResponse:
	"""Test the ModelLoadProgressResponse schema validation."""

	def test_schema_validation_with_required_fields(self):
		# Arrange
		from app.cores.model_loader.schemas import ModelLoadPhase, ModelLoadProgressResponse

		# Act
		response = ModelLoadProgressResponse(
			id='test-model',
			step=5,
			phase=ModelLoadPhase.LOADING_MODEL,
			message='Loading model weights...',
		)

		# Assert
		assert response.id == 'test-model'
		assert response.step == 5
		assert response.total == 9  # default value
		assert response.phase == ModelLoadPhase.LOADING_MODEL
		assert response.message == 'Loading model weights...'

	def test_schema_validation_with_custom_total(self):
		# Arrange
		from app.cores.model_loader.schemas import ModelLoadPhase, ModelLoadProgressResponse

		# Act
		response = ModelLoadProgressResponse(
			id='test-model', step=3, total=10, phase=ModelLoadPhase.INITIALIZATION, message='Test message'
		)

		# Assert
		assert response.total == 10

	def test_schema_serialization(self):
		# Arrange
		from app.cores.model_loader.schemas import ModelLoadPhase, ModelLoadProgressResponse

		response = ModelLoadProgressResponse(
			id='test-model', step=5, phase=ModelLoadPhase.LOADING_MODEL, message='Loading model weights...'
		)

		# Act
		serialized = response.model_dump()

		# Assert
		assert serialized == {
			'id': 'test-model',
			'step': 5,
			'total': 9,
			'phase': 'loading_model',
			'message': 'Loading model weights...',
		}


class TestEmitProgress:
	"""Test the emit_progress helper function."""

	def test_emit_progress_calls_socket_service(self, mock_dependencies):
		# Arrange
		from app.cores.model_loader.schemas import ModelLoadPhase

		emit_progress = mock_dependencies['emit_progress']
		mock_socket_service = mock_dependencies['mock_socket_service']

		# Act
		emit_progress('test-model', 5, 'Loading model weights...')

		# Assert
		mock_socket_service.model_load_progress.assert_called_once()
		call_args = mock_socket_service.model_load_progress.call_args[0][0]
		assert call_args.id == 'test-model'
		assert call_args.step == 5
		assert call_args.total == 9
		assert call_args.phase == ModelLoadPhase.LOADING_MODEL
		assert call_args.message == 'Loading model weights...'

	def test_emit_progress_logs_structured_message(self, mock_dependencies):
		# Arrange
		emit_progress = mock_dependencies['emit_progress']
		mock_logger = mock_dependencies['mock_logger']

		# Act
		emit_progress('test-model', 3, 'Checking model cache...')

		# Assert
		mock_logger.info.assert_called_once()
		log_message = mock_logger.info.call_args[0][0]
		assert '[ModelLoad]' in log_message
		assert 'test-model' in log_message
		assert 'step=3/9' in log_message
		assert 'phase=loading_model' in log_message
		assert 'msg="Checking model cache..."' in log_message

	def test_emit_progress_handles_socket_failure_gracefully(self, mock_dependencies):
		# Arrange
		emit_progress = mock_dependencies['emit_progress']
		mock_socket_service = mock_dependencies['mock_socket_service']
		mock_logger = mock_dependencies['mock_logger']
		mock_socket_service.model_load_progress.side_effect = RuntimeError('Socket failed')

		# Act - should not raise exception
		emit_progress('test-model', 5, 'Loading model weights...')

		# Assert - warning should be logged
		mock_logger.warning.assert_called_once()
		warning_message = mock_logger.warning.call_args[0][0]
		assert 'Failed to emit model load progress' in warning_message


class TestModelLoadProgressIntegration:
	"""Integration tests for model load progress functionality."""

	def test_model_loader_emits_start_event(self, mock_dependencies):
		# Arrange
		model_loader = mock_dependencies['model_loader']
		mock_socket_service = mock_dependencies['mock_socket_service']
		model_id = 'test-model'

		# Act
		_ = model_loader(model_id)

		# Assert - start event should be emitted first
		calls = mock_socket_service.method_calls
		start_calls = [c for c in calls if c[0] == 'model_load_started']
		assert len(start_calls) == 1
		assert start_calls[0][1][0].id == model_id

	def test_model_loader_emits_all_9_progress_events(self, mock_dependencies):
		# Arrange
		model_loader = mock_dependencies['model_loader']
		mock_socket_service = mock_dependencies['mock_socket_service']
		model_id = 'test-model'

		# Act
		_ = model_loader(model_id)

		# Assert - should have 9 progress emissions
		calls = mock_socket_service.method_calls
		progress_calls = [c for c in calls if c[0] == 'model_load_progress']
		assert len(progress_calls) == 9

		# Verify steps are in order (1-9)
		steps = [c[1][0].step for c in progress_calls]
		assert steps == [1, 2, 3, 4, 5, 6, 7, 8, 9]

	def test_model_loader_emits_events_in_correct_order(self, mock_dependencies):
		# Arrange
		model_loader = mock_dependencies['model_loader']
		mock_socket_service = mock_dependencies['mock_socket_service']
		model_id = 'test-model'

		# Act
		_ = model_loader(model_id)

		# Assert - order should be: START → PROGRESS(1-9) → COMPLETED
		calls = mock_socket_service.method_calls
		event_sequence = [c[0] for c in calls]

		# Find indices of key events
		start_idx = event_sequence.index('model_load_started')
		first_progress_idx = event_sequence.index('model_load_progress')
		completed_idx = event_sequence.index('model_load_completed')

		# Verify order
		assert start_idx < first_progress_idx < completed_idx

	def test_model_loader_progress_has_correct_phases(self, mock_dependencies):
		# Arrange
		from app.cores.model_loader.schemas import ModelLoadPhase

		model_loader = mock_dependencies['model_loader']
		mock_socket_service = mock_dependencies['mock_socket_service']
		model_id = 'test-model'

		# Act
		_ = model_loader(model_id)

		# Assert - verify phase mapping for each step
		calls = mock_socket_service.method_calls
		progress_calls = [c for c in calls if c[0] == 'model_load_progress']

		expected_phases = {
			1: ModelLoadPhase.INITIALIZATION,
			2: ModelLoadPhase.INITIALIZATION,
			3: ModelLoadPhase.LOADING_MODEL,
			4: ModelLoadPhase.LOADING_MODEL,
			5: ModelLoadPhase.LOADING_MODEL,
			6: ModelLoadPhase.DEVICE_SETUP,
			7: ModelLoadPhase.DEVICE_SETUP,
			8: ModelLoadPhase.OPTIMIZATION,
			9: ModelLoadPhase.OPTIMIZATION,
		}

		for progress_call in progress_calls:
			progress_data = progress_call[1][0]
			assert progress_data.phase == expected_phases[progress_data.step]

	def test_model_loader_progress_has_correct_messages(self, mock_dependencies):
		# Arrange
		model_loader = mock_dependencies['model_loader']
		mock_socket_service = mock_dependencies['mock_socket_service']
		model_id = 'test-model'

		# Act
		_ = model_loader(model_id)

		# Assert - verify messages for each step
		calls = mock_socket_service.method_calls
		progress_calls = [c for c in calls if c[0] == 'model_load_progress']

		expected_messages = {
			1: 'Initializing model loader...',
			2: 'Loading feature extractor...',
			3: 'Checking model cache...',
			4: 'Preparing loading strategies...',
			5: 'Loading model weights...',
			6: 'Model loaded successfully',
			7: 'Moving model to device...',
			8: 'Applying optimizations...',
			9: 'Finalizing model setup...',
		}

		for progress_call in progress_calls:
			progress_data = progress_call[1][0]
			assert progress_data.message == expected_messages[progress_data.step]

	def test_model_loader_continues_on_progress_emission_failure(self, mock_dependencies):
		# Arrange
		model_loader = mock_dependencies['model_loader']
		mock_socket_service = mock_dependencies['mock_socket_service']
		mock_pipe = mock_dependencies['mock_pipe']
		model_id = 'test-model'

		# Make progress emission fail
		mock_socket_service.model_load_progress.side_effect = RuntimeError('Socket failed')

		# Act - should not raise exception
		result = model_loader(model_id)

		# Assert - model should still load successfully
		assert result == mock_pipe
		# Completed event should still be emitted (it doesn't use emit_progress)
		mock_socket_service.model_load_completed.assert_called_once()
