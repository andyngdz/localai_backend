import asyncio
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock

import pytest

from app.socket.schemas import SocketEvents
from app.socket.service import SocketService


class PayloadModel:
    """Lightweight Pydantic-like object exposing model_dump()."""

    def __init__(self, **data: Any) -> None:
        self._data = data

    def model_dump(self) -> Dict[str, Any]:
        return dict(self._data)


@pytest.mark.asyncio
async def test_emit_calls_sio_emit(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    service = SocketService()
    mock_emit = AsyncMock(return_value=None)
    monkeypatch.setattr(service.sio, "emit", mock_emit)
    event = SocketEvents.DOWNLOAD_START
    data = {"foo": "bar"}

    # Act
    await service.emit(event, data)

    # Assert
    mock_emit.assert_awaited_once_with(event, data=data)
    # Close the custom loop created by SocketService to avoid unclosed loop warnings
    try:
        service.loop.close()
    except Exception:
        pass


def test_emit_sync_schedules_coroutine_threadsafe(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    test_loop = asyncio.new_event_loop()
    monkeypatch.setattr(asyncio, "get_event_loop", lambda: test_loop)
    service = SocketService()
    mock_emit = AsyncMock(return_value=None)
    monkeypatch.setattr(service.sio, "emit", mock_emit)

    captured_args = {}

    def fake_run_coroutine_threadsafe(coro: Any, loop: asyncio.AbstractEventLoop) -> Mock:
        captured_args["coro"] = coro
        captured_args["loop"] = loop
        # Return a dummy Future-like object
        # Ensure the coroutine is executed to avoid "never awaited" warnings
        loop.run_until_complete(coro)
        fut = Mock()
        fut.cancelled.return_value = False
        fut.done.return_value = True
        return fut

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", fake_run_coroutine_threadsafe)

    event = SocketEvents.DOWNLOAD_COMPLETED
    data = {"ok": True}

    # Act
    service.emit_sync(event, data)

    # Assert
    mock_emit.assert_called_once_with(event, data=data)
    assert "coro" in captured_args and asyncio.iscoroutine(captured_args["coro"]) is True
    assert captured_args["loop"] is service.loop
    test_loop.close()


@pytest.mark.asyncio
async def test_download_start_emits_correct_event(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    service = SocketService()
    mock_emit = AsyncMock(return_value=None)
    monkeypatch.setattr(service, "emit", mock_emit)
    payload = PayloadModel(model="m1", size=123)

    # Act
    await service.download_start(payload)

    # Assert
    mock_emit.assert_awaited_once_with(SocketEvents.DOWNLOAD_START, data=payload.model_dump())
    # Close the custom loop created by SocketService to avoid unclosed loop warnings
    try:
        service.loop.close()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_download_completed_emits_correct_event(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    service = SocketService()
    mock_emit = AsyncMock(return_value=None)
    monkeypatch.setattr(service, "emit", mock_emit)
    payload = PayloadModel(total=10, succeeded=10)

    # Act
    await service.download_completed(payload)

    # Assert
    mock_emit.assert_awaited_once_with(
        SocketEvents.DOWNLOAD_COMPLETED, data=payload.model_dump()
    )
    # Close the custom loop created by SocketService to avoid unclosed loop warnings
    try:
        service.loop.close()
    except Exception:
        pass


def test_download_step_progress_emits_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    test_loop = asyncio.new_event_loop()
    monkeypatch.setattr(asyncio, "get_event_loop", lambda: test_loop)
    service = SocketService()
    mock_emit_sync = Mock()
    monkeypatch.setattr(service, "emit_sync", mock_emit_sync)
    payload = PayloadModel(step=3, of=10)

    # Act
    service.download_step_progress(payload)

    # Assert
    mock_emit_sync.assert_called_once_with(
        SocketEvents.DOWNLOAD_STEP_PROGRESS, data=payload.model_dump()
    )
    test_loop.close()


def test_model_load_completed_emits_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    test_loop = asyncio.new_event_loop()
    monkeypatch.setattr(asyncio, "get_event_loop", lambda: test_loop)
    service = SocketService()
    mock_emit_sync = Mock()
    monkeypatch.setattr(service, "emit_sync", mock_emit_sync)
    payload = PayloadModel(model="m1", status="ready")

    # Act
    service.model_load_completed(payload)

    # Assert
    mock_emit_sync.assert_called_once_with(
        SocketEvents.MODEL_LOAD_COMPLETED, data=payload.model_dump()
    )
    test_loop.close()


def test_image_generation_step_end_emits_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    test_loop = asyncio.new_event_loop()
    monkeypatch.setattr(asyncio, "get_event_loop", lambda: test_loop)
    service = SocketService()
    mock_emit_sync = Mock()
    monkeypatch.setattr(service, "emit_sync", mock_emit_sync)
    payload = PayloadModel(step=5)

    # Act
    service.image_generation_step_end(payload)

    # Assert
    mock_emit_sync.assert_called_once_with(
        SocketEvents.IMAGE_GENERATION_STEP_END, data=payload.model_dump()
    )
    test_loop.close()
