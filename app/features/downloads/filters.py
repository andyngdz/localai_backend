import fnmatch
from pathlib import PurePosixPath
from typing import List


def get_directory_from_path(file_path: str) -> str:
	"""Extract directory path from a file path."""
	parent = str(PurePosixPath(file_path).parent)
	return parent if parent != '.' else ''


def get_filename_from_path(file_path: str) -> str:
	"""Extract filename from a file path."""
	return PurePosixPath(file_path).name


def get_ignore_components(files: List[str], scopes: List[str]) -> List[str]:
	"""
	Return files that should be ignored to avoid downloading bloat.

	Strategy:
	1. Keep only STANDARD .safetensors files (not fp16/non_ema/ema_only)
	2. Filter duplicate .bin files (when standard .safetensors exists in same directory)
	3. Filter all variants (fp16, non_ema, ema_only)

	This reduces download size from ~10-15 GB to ~4.3 GB for typical models.

	Example:
		- "unet/diffusion_pytorch_model.safetensors" → kept (standard)
		- "unet/diffusion_pytorch_model.bin" → ignored (duplicate of .safetensors)
		- "unet/diffusion_pytorch_model.fp16.safetensors" → ignored (variant)
		- "unet/diffusion_pytorch_model.non_ema.safetensors" → ignored (training artifact)
	"""
	in_scope = [file_path for file_path in files if any(fnmatch.fnmatch(file_path, scope) for scope in scopes)]

	ignored = []

	dirs_with_standard_safetensors = set()
	for file_path in in_scope:
		if file_path.endswith('.safetensors'):
			directory = get_directory_from_path(file_path)
			if directory:
				filename = get_filename_from_path(file_path)
				if not any(variant in filename for variant in ['fp16', 'non_ema', 'ema_only']):
					dirs_with_standard_safetensors.add(directory)

	for file_path in in_scope:
		if file_path.endswith('.bin'):
			directory = get_directory_from_path(file_path)
			if directory in dirs_with_standard_safetensors:
				ignored.append(file_path)

	for file_path in in_scope:
		if file_path not in ignored:
			filename = get_filename_from_path(file_path)
			if any(variant in filename for variant in ['fp16', 'non_ema', 'ema_only']):
				ignored.append(file_path)

	return ignored
