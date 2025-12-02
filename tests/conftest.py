"""
Global pytest configuration for the LocalAI backend tests.

This file is automatically loaded by pytest before any tests are run.
"""

import pathlib
import sys
import types

_FUNCTIONAL_TENSOR_MODULE = 'torchvision.transforms.functional_tensor'


def _patch_torchvision_functional_tensor():
	"""
	Patch for torchvision.transforms.functional_tensor compatibility.

	The module was removed in torchvision 0.17+.
	Some dependencies (like basicsr) still import from this deprecated path.
	This creates a shim module that redirects to the new location.
	"""
	if _FUNCTIONAL_TENSOR_MODULE in sys.modules:
		return

	try:
		__import__(_FUNCTIONAL_TENSOR_MODULE)
		return
	except ImportError:
		pass

	from torchvision.transforms import functional

	functional_tensor = types.ModuleType(_FUNCTIONAL_TENSOR_MODULE)
	setattr(functional_tensor, 'rgb_to_grayscale', functional.rgb_to_grayscale)
	sys.modules[_FUNCTIONAL_TENSOR_MODULE] = functional_tensor


# Apply patch before any imports
_patch_torchvision_functional_tensor()


# Add project root to Python path for all tests
# This is a cleaner approach than modifying sys.path in individual test files
def pytest_configure(config):
	"""
	Pytest hook that runs before tests are collected.

	This ensures the project root is in sys.path so that tests can import
	from the app package directly without relative imports.
	"""
	project_root = pathlib.Path(__file__).resolve().parent.parent
	if str(project_root) not in sys.path:
		sys.path.insert(0, str(project_root))
