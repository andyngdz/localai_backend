"""Service for applying third-party package patches at startup."""

import asyncio
import sys
import types

from torchvision.transforms.functional import rgb_to_grayscale

# Windows-specific event loop policy - only available on Windows
# Using sys.platform here for type narrowing at module level
if sys.platform == 'win32':
	WindowsProactorEventLoopPolicy = asyncio.WindowsProactorEventLoopPolicy
else:
	WindowsProactorEventLoopPolicy = None


def init() -> None:
	"""Apply all necessary patches for third-party package compatibility.

	Must be called early in application startup, before any patched packages are imported.
	"""
	_patch_basicsr_torchvision()


def setup_windows_event_loop() -> None:
	"""Set Windows-specific event loop policy for compatibility."""
	if WindowsProactorEventLoopPolicy is None:
		return

	asyncio.set_event_loop_policy(WindowsProactorEventLoopPolicy())


def _patch_basicsr_torchvision() -> None:
	"""Patch basicsr's deprecated torchvision import.

	basicsr uses `torchvision.transforms.functional_tensor.rgb_to_grayscale` which was
	removed in torchvision 0.18+. Create a shim module with the expected function.
	"""
	_functional_tensor = types.ModuleType('torchvision.transforms.functional_tensor')
	setattr(_functional_tensor, 'rgb_to_grayscale', rgb_to_grayscale)
	sys.modules['torchvision.transforms.functional_tensor'] = _functional_tensor
