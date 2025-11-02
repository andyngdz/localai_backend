"""Tests for platform optimizer factory."""

from unittest.mock import patch

import pytest


class TestGetOptimizer:
	"""Test get_optimizer factory function."""

	@patch('app.cores.platform_optimizations.factory.sys')
	def test_returns_windows_optimizer_on_win32(self, mock_sys):
		"""Test returns WindowsOptimizer on Windows."""
		from app.cores.platform_optimizations.factory import get_optimizer
		from app.cores.platform_optimizations.windows import WindowsOptimizer

		mock_sys.platform = 'win32'

		optimizer = get_optimizer()

		assert isinstance(optimizer, WindowsOptimizer)
		assert optimizer.get_platform_name() == 'Windows'

	@patch('app.cores.platform_optimizations.factory.sys')
	def test_returns_linux_optimizer_on_linux(self, mock_sys):
		"""Test returns LinuxOptimizer on Linux."""
		from app.cores.platform_optimizations.factory import get_optimizer
		from app.cores.platform_optimizations.linux import LinuxOptimizer

		mock_sys.platform = 'linux'

		optimizer = get_optimizer()

		assert isinstance(optimizer, LinuxOptimizer)
		assert optimizer.get_platform_name() == 'Linux'

	@patch('app.cores.platform_optimizations.factory.sys')
	def test_returns_darwin_optimizer_on_macos(self, mock_sys):
		"""Test returns DarwinOptimizer on macOS."""
		from app.cores.platform_optimizations.factory import get_optimizer
		from app.cores.platform_optimizations.darwin import DarwinOptimizer

		mock_sys.platform = 'darwin'

		optimizer = get_optimizer()

		assert isinstance(optimizer, DarwinOptimizer)
		assert optimizer.get_platform_name() == 'macOS'

	@patch('app.cores.platform_optimizations.factory.sys')
	def test_raises_runtime_error_on_unsupported_platform(self, mock_sys):
		"""Test raises RuntimeError for unsupported platforms."""
		from app.cores.platform_optimizations.factory import get_optimizer

		mock_sys.platform = 'freebsd'

		with pytest.raises(RuntimeError, match='Unsupported platform: freebsd'):
			get_optimizer()

	@patch('app.cores.platform_optimizations.factory.sys')
	def test_raises_runtime_error_on_unknown_platform(self, mock_sys):
		"""Test raises RuntimeError for completely unknown platforms."""
		from app.cores.platform_optimizations.factory import get_optimizer

		mock_sys.platform = 'unknown_os'

		with pytest.raises(RuntimeError, match='Unsupported platform: unknown_os'):
			get_optimizer()
