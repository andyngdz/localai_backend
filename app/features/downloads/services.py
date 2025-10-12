import asyncio
import fnmatch
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List

from huggingface_hub import HfApi, hf_hub_download
from sqlalchemy.orm import Session
from tqdm import tqdm as BaseTqdm

from app.services.models import model_service
from app.socket import socket_service
from config import CACHE_FOLDER

from .schemas import DownloadStepProgressResponse

logger = logging.getLogger(__name__)


class DownloadTqdm(BaseTqdm):
	"""Custom tqdm for overall snapshot progress."""

	def __init__(self, *args, **kwargs):
		self.id = kwargs.pop('id')
		self.desc = kwargs.pop('desc')
		self.file_sizes = kwargs.pop('file_sizes')
		self.downloaded_size = 0
		self.total_downloaded_size = sum(self.file_sizes)
		# Use auto-disable so internal counters (self.n) still update in non-TTY contexts
		# When disable=True, tqdm.update() short-circuits and does not increment self.n
		kwargs.setdefault('disable', None)
		kwargs.setdefault('file', sys.stderr)
		kwargs.setdefault('mininterval', 0.25)
		kwargs.setdefault('leave', False)
		super().__init__(*args, **kwargs)

	def set_file_size(self, index: int, size: int) -> None:
		"""Update the recorded size for a file and adjust totals accordingly."""
		if index < 0 or index >= len(self.file_sizes):
			return

		size = max(size, 0)
		previous = self.file_sizes[index]

		if previous == size:
			return

		self.file_sizes[index] = size
		self.total_downloaded_size += size - previous

	def update(self, n=1):
		prev_n = self.n
		super().update(n)

		for index in range(prev_n, min(self.n, len(self.file_sizes))):
			self.downloaded_size += self.file_sizes[index]

		total = self.total
		desc = self.desc

		socket_service.download_step_progress(
			DownloadStepProgressResponse(
				id=self.id,
				step=self.n,
				total=total,
				downloaded_size=self.downloaded_size,
				total_downloaded_size=self.total_downloaded_size,
			)
		)

		logger.info(f'{desc} {self.n}/{total}')

	def close(self):
		super().close()


class DownloadService:
	def __init__(self):
		self.executor = ThreadPoolExecutor()
		self.api = HfApi()

	async def start(self, id: str, db: Session):
		loop = asyncio.get_event_loop()
		local_dir = await loop.run_in_executor(
			self.executor, 
			self.download_model, 
			id, 
			db
		)
		return local_dir

	def download_model(self, id: str, db: Session):
		components = self.get_components(id)
		components_scopes = [f'{c}/*' for c in components]
		files = self.list_files(id)
		ignore_components = self.get_ignore_components(files, components_scopes)
		file_sizes_map = self.get_file_sizes_map(id)

		files_to_download = [
			f
			for f in files
			if (f == 'model_index.json' or any(fnmatch.fnmatch(f, p) for p in components_scopes))
			and f not in ignore_components
		]

		# Download model_index.json first to anchor snapshot root
		files_to_download.sort(key=lambda f: (f != 'model_index.json', f))
		file_sizes = [file_sizes_map.get(filename, 0) for filename in files_to_download]

		total = len(files_to_download)
		if total == 0:
			logger.warning('No files to download')
			return

		logger.info(f'Starting download of {total} files for model {id}')
		progress = DownloadTqdm(
			id=id,
			total=total,
			desc=f'Downloading {id}',
			unit='files',
			file_sizes=file_sizes,
		)

		local_dir = None

		try:
			for index, filename in enumerate(files_to_download, start=1):
				local_path = hf_hub_download(
					repo_id=id,
					filename=filename,
					cache_dir=CACHE_FOLDER,
				)

				try:
					file_size = os.path.getsize(local_path)
				except OSError:
					file_size = file_sizes_map.get(filename, 0)

				progress.set_file_size(index - 1, file_size)
				if local_dir is None:
					# If we downloaded model_index.json first, this will be the snapshot root
					local_dir = os.path.dirname(local_path)
				progress.update(1)
		finally:
			progress.close()

		logger.info(f'All files downloaded to {local_dir}')

		# Save the downloaded model to the database
		if local_dir:
			try:
				model_service.add_model(db, id, local_dir)
				logger.info(f'Model {id} saved to database with path {local_dir}')
			except Exception as error:
				logger.error(f'Failed to save model {id} to database: {error}')

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

		if not info.siblings:
			return []

		return [s.rfilename for s in info.siblings]

	def get_file_sizes_map(self, id: str):
		info = self.api.repo_info(id)

		if not info.siblings:
			return {}

		return {
			s.rfilename: getattr(s, 'size', 0) or 0
			for s in info.siblings
		}

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
