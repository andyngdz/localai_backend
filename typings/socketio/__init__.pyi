"""Type stubs for python-socketio library.

This module provides type hints for the socketio library APIs used in this project.
"""

from typing import Any, Literal

class AsyncServer:
	"""Socket.IO AsyncServer for ASGI applications."""

	def __init__(
		self,
		*,
		async_mode: Literal['asgi'] = 'asgi',
		cors_allowed_origins: str | list[str] = '*',
		logger: bool = False,
		**kwargs: Any,
	) -> None:
		"""Initialize AsyncServer with ASGI mode."""
		...

	async def emit(
		self,
		event: str,
		data: dict[str, Any] | None = None,
		**kwargs: Any,
	) -> None:
		"""Emit an event to all connected clients."""
		...

class ASGIApp:
	"""ASGI application wrapper for Socket.IO server."""

	def __init__(self, socketio_server: AsyncServer, **kwargs: Any) -> None:
		"""Wrap AsyncServer as ASGI application."""
		...
