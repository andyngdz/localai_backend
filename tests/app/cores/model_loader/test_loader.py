from __future__ import annotations

from unittest.mock import ANY, MagicMock, patch

import pytest

from app.cores.model_loader.model_loader import model_loader, move_to_device


@pytest.fixture
def mock_dependencies():
    with patch('app.cores.model_loader.model_loader.SessionLocal') as mock_session, \
         patch('app.cores.model_loader.model_loader.MaxMemoryConfig') as mock_max_memory, \
         patch('app.cores.model_loader.model_loader.device_service') as mock_device_service, \
         patch('app.cores.model_loader.model_loader.CLIPImageProcessor') as mock_clip_processor, \
         patch('app.cores.model_loader.model_loader.StableDiffusionSafetyChecker') as mock_safety_checker, \
         patch('app.cores.model_loader.model_loader.AutoPipelineForText2Image') as mock_auto_pipeline, \
         patch('app.cores.model_loader.model_loader.socket_service') as mock_socket_service, \
         patch('app.cores.model_loader.model_loader.logger') as mock_logger:

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
        }


def test_model_loader_success(mock_dependencies):
    # Arrange
    model_id = 'test-model-id'
    mock_pipe = mock_dependencies['mock_pipe']
    mock_auto_pipeline = mock_dependencies['mock_auto_pipeline']
    mock_socket_service = mock_dependencies['mock_socket_service']

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


def test_move_to_device_to_empty_success():
    # Arrange
    mock_pipe = MagicMock()
    device = 'cpu'
    log_prefix = 'Test model'
    
    # Act
    result = move_to_device(mock_pipe, device, log_prefix)
    
    # Assert
    assert result == mock_pipe.to_empty.return_value
    mock_pipe.to_empty.assert_called_once_with(device)
    mock_pipe.to.assert_not_called()


def test_move_to_device_fallback_to_to():
    # Arrange
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


def test_move_to_device_fallback_to_to_type_error():
    # Arrange
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
