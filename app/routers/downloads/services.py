import fnmatch
import json
import logging
import os
import sys
from typing import List

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm as BaseTqdm

from config import CACHE_FOLDER

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = 1
logger = logging.getLogger(__name__)


class DownloadTqdm(BaseTqdm):
	"""Custom tqdm for overall snapshot progress."""

	def __init__(self, *args, **kwargs):
		kwargs.setdefault('file', sys.stderr)
		kwargs.setdefault('mininterval', 0.25)
		kwargs.setdefault('leave', False)
		super().__init__(*args, **kwargs)

	def update(self, n=1):
		super().update(n)
		total = self.total
		desc = self.desc
		logger.info(f'{desc} {self.n}/{total}')

	def close(self):
		try:
			print('', flush=True)
		finally:
			return super().close()


class DownloadService:
	api = HfApi()

	def download_model(self, id: str):
		components = self.get_components(id)
		components_scopes = [f'{c}/*' for c in components]
		files = self.list_files(id)
		ignore_components = self.get_ignore_components(files, components_scopes)

		files_to_download = [
			f
			for f in files
			if (f == 'model_index.json' or any(fnmatch.fnmatch(f, p) for p in components_scopes))
			and f not in ignore_components
		]

		# Download model_index.json first to anchor snapshot root
		files_to_download.sort(key=lambda f: (f != 'model_index.json', f))

		total = len(files_to_download)
		if total == 0:
			logger.error('No files to download')
			return

		logger.info(f'\n[Download] Starting download of {total} files for model {id}')
		progress = DownloadTqdm(total=total, desc=f'Downloading {id}', unit='files')

		local_dir = None

		try:
			for index, filename in enumerate(files_to_download, start=1):
				local_path = hf_hub_download(
					repo_id=id,
					filename=filename,
					cache_dir=CACHE_FOLDER,
				)
				if local_dir is None:
					# If we downloaded model_index.json first, this will be the snapshot root
					local_dir = os.path.dirname(local_path)
				progress.update(1)
		finally:
			progress.close()

		logger.info(f'[Download] All files downloaded to {local_dir}')

		return local_dir

	def get_ignore_components(self, files: List[str], scopes: List[str]):
		"""
		Return a list of .bin files that should be ignored if a corresponding
		.safetensors file with the same base name exists in the given scopes.

		Example:
			- If both "unet/model.bin" and "unet/model.safetensors" exist,
				then "unet/model.bin" will be ignored.
			- If only "vae/model.bin" exists (no safetensors), it will be kept.
		"""
		# Filter files that are inside the provided scopes (e.g. "unet/*", "vae/*")
		in_scope = [f for f in files if any(fnmatch.fnmatch(f, p) for p in scopes)]

		# Collect base names of all safetensors files in scope
		safetensors_bases = {f.removesuffix('.safetensors') for f in in_scope if f.endswith('.safetensors')}

		# Return .bin files that have a matching .safetensors base
		return [f for f in in_scope if f.endswith('.bin') and f.removesuffix('.bin') in safetensors_bases]

	def list_files(self, id: str):
		info = self.api.repo_info(id)

		return [s.rfilename for s in info.siblings]

	def get_components(self, id: str):
		model_index = hf_hub_download(
			repo_id=id,
			filename='model_index.json',
			repo_type='model',
		)

		with open(model_index, 'r', encoding='utf-8') as f:
			data = json.load(f)

		components = []

		for key, val in data.items():
			if isinstance(val, list) and val[0] is not None:
				components.append(key)

		return components


download_service = DownloadService()
