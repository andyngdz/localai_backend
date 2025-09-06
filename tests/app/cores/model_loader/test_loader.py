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
        setattr(diffusers_stub, name, type(name, (), {}) )

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

        mock_pipe = MagicMock()
        mock_auto_pipeline.from_pretrained.return_value = mock_pipe
        # Set up to_empty to return the same mock_pipe
        mock_pipe.to_empty.return_value = mock_pipe
        # Set up to as fallback
        mock_pipe.to.return_value = mock_pipe

        yield {
            'mock_session': mock_session,
            'mock_max_memory': mock_max_memory,
            'mock_device_service': mock_device_service,
            'mock_clip_processor': mock_clip_processor,
            'mock_safety_checker': mock_safety_checker,
            'mock_auto_pipeline': mock_auto_pipeline,
            'mock_socket_service': mock_socket_service,
            'mock_logger': mock_logger,
            'mock_pipe': mock_pipe,
            'module': target_module,
            'model_loader': getattr(target_module, 'model_loader'),
            'move_to_device': getattr(target_module, 'move_to_device'),
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

    # First call raises EnvironmentError, second returns pipe (fallback path)
    call_sequence: list[MagicMock | Exception] = [EnvironmentError('safetensors not available'), mock_pipe]

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
    # Verify the second call used use_safetensors=False
    _, second_kwargs = mock_auto_pipeline.from_pretrained.call_args
    assert second_kwargs.get('use_safetensors') is False
    mock_socket_service.model_load_completed.assert_called_once()


def test_model_loader_environment_error_then_failure(mock_dependencies):
    # Arrange
    model_id = 'test-model-id'
    mock_auto_pipeline = mock_dependencies['mock_auto_pipeline']
    mock_socket_service = mock_dependencies['mock_socket_service']
    model_loader = mock_dependencies['model_loader']

    # First call raises EnvironmentError, second raises a generic Exception
    def side_effect(*args, **kwargs):
        if mock_auto_pipeline.from_pretrained.call_count == 0:
            raise EnvironmentError('first attempt failed')
        raise RuntimeError('second attempt failed')

    mock_auto_pipeline.from_pretrained.side_effect = side_effect

    # Act & Assert
    with pytest.raises(RuntimeError, match='second attempt failed'):
        model_loader(model_id)

    # Ensure failure was emitted with correct payload
    mock_socket_service.model_load_failed.assert_called_once()
    args, _ = mock_socket_service.model_load_failed.call_args
    assert args[0].id == model_id
    assert args[0].error == 'second attempt failed'


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
