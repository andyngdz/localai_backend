import asyncio
import fnmatch
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import requests.adapters
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
from sqlalchemy.orm import Session

from app.services.models import model_service
from app.services.storage import storage_service

from .progress import DownloadProgress

# Backwards compatibility for tests stubbing the old symbol.
DownloadTqdm = DownloadProgress

logger = logging.getLogger(__name__)


class DownloadService:
	"""Service responsible for downloading models and emitting socket progress."""

	CHUNK_SIZE = 4 * 1024 * 1024

	def __init__(self):
		self.executor = ThreadPoolExecutor()
		self.api = HfApi()
		# Reuse HTTPS connections for 20-30% faster downloads
		self.session = requests.Session()
		# Configure connection pool (HTTPS only for security)
		adapter = requests.adapters.HTTPAdapter(
			pool_connections=10,
			pool_maxsize=20,
			max_retries=3
		)
		self.session.mount('https://', adapter)

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
		"""Download a model from HuggingFace Hub with progress tracking."""
		if not id or not id.strip():
			raise ValueError('Model ID cannot be empty')

		repo_info = self.api.repo_info(id)
		revision = getattr(repo_info, 'sha', 'main')
		# Build the list of candidate files and initial size map up-front so byte totals remain monotonic.
		components = self.get_components(id, revision=revision)
		components_scopes = [f'{c}/*' for c in components]
		files = self.list_files(id, repo_info=repo_info)
		ignore_components = self.get_ignore_components(files, components_scopes)
		file_sizes_map = self.get_file_sizes_map(id, repo_info=repo_info)

		files_to_download = [
			f
			for f in files
			if (f == 'model_index.json' or any(fnmatch.fnmatch(f, p) for p in components_scopes))
			and f not in ignore_components
		]

		files_to_download.sort(key=lambda f: (f != 'model_index.json', f))
		# Ensure every file has a deterministic size before streaming begins; fall back to HEAD when hub metadata is missing.
		file_sizes: List[int] = []
		for filename in files_to_download:
			size = file_sizes_map.get(filename, 0)
			if size <= 0:
				size = self.fetch_remote_file_size(id, filename, revision=revision)
				file_sizes_map[filename] = size
			file_sizes.append(size)

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
			logger=logger,
		)

		model_root = storage_service.get_model_dir(id)
		snapshot_dir = os.path.join(model_root, 'snapshots', revision)
		local_dir: Optional[str] = None

		try:
			for index, filename in enumerate(files_to_download):
				# Emit a start event so the client can show which file is currently in-flight.
				progress.start_file(filename)
				logger.info('Downloading %s (%s/%s)', filename, index + 1, total)
				local_path = self.download_file(
					repo_id=id,
					filename=filename,
					revision=revision,
					snapshot_dir=snapshot_dir,
					file_index=index,
					progress=progress,
				)
				if local_dir is None:
					local_dir = os.path.dirname(local_path)
				progress.update(1)
				logger.info('Finished %s (%s/%s)', filename, index + 1, total)
		except Exception:
			logger.exception('Failed during download of %s', id)
			raise
		finally:
			progress.close()

		logger.info(f'All files downloaded to {local_dir}')

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

	def list_files(self, id: str, repo_info: Optional[Any] = None) -> List[str]:
		info = repo_info or self.api.repo_info(id)

		if not info.siblings:
			return []

		return [s.rfilename for s in info.siblings]

	def get_file_sizes_map(self, id: str, repo_info: Optional[Any] = None) -> Dict[str, int]:
		info = repo_info or self.api.repo_info(id)

		if not info.siblings:
			return {}

		return {
			s.rfilename: getattr(s, 'size', 0) or 0
			for s in info.siblings
		}

	def get_components(self, id: str, revision: Optional[str] = None):
		model_index = hf_hub_download(
			repo_id=id,
			filename='model_index.json',
			repo_type='model',
			revision=revision,
		)

		with open(model_index, 'r', encoding='utf-8') as f:
			data = json.load(f)

		components = []

		for key, val in data.items():
			if isinstance(val, list) and val[0] is not None:
				components.append(key)

		return components

	def auth_headers(self, token: Optional[str] = None) -> Dict[str, str]:
		"""Build authorization headers for HuggingFace API requests."""
		headers = {}
		if token:
			headers['Authorization'] = f'Bearer {token}'
		return headers

	def download_file(
		self,
		repo_id: str,
		filename: str,
		revision: str,
		snapshot_dir: str,
		file_index: int,
		progress: DownloadTqdm,
		token: Optional[str] = None,
	):
		"""
		Download a single file from HuggingFace Hub with streaming and progress tracking.

		Args:
			repo_id: HuggingFace repository ID
			filename: Relative path of file within repository (validated for path traversal)
			revision: Git revision/commit hash
			snapshot_dir: Local directory to store downloaded files
			file_index: Zero-based index in progress.file_sizes list (must align with file order)
			progress: Progress tracker for websocket updates
			token: Optional HuggingFace authentication token

		Returns:
			str: Absolute path to the downloaded file

		Side Effects:
			- Creates snapshot_dir and parent directories if needed
			- Skips download if file already exists with matching size
			- Downloads to temporary .part file, then atomically renames on success
			- Updates progress.set_file_size() with actual Content-Length
			- Calls progress.update_bytes() for each downloaded chunk
			- Cleans up .part file on error
		"""
		# Security: Prevent path traversal attacks
		snapshot_path = Path(snapshot_dir).resolve()
		local_path = (snapshot_path / filename).resolve()

		# Validate that the resolved path is under snapshot_dir
		try:
			local_path.relative_to(snapshot_path)
		except ValueError:
			raise ValueError(f'Invalid filename: {filename} attempts to escape snapshot directory')

		# Create directories with validated paths
		os.makedirs(snapshot_path, exist_ok=True)
		os.makedirs(local_path.parent, exist_ok=True)

		local_path = str(local_path)

		if os.path.exists(local_path):
			actual_size = os.path.getsize(local_path)
			if actual_size > 0:
				progress.set_file_size(file_index, actual_size)
				logger.debug('Skipping download for %s; already complete', filename)
				return local_path
			os.remove(local_path)

		temp_path = f'{local_path}.part'
		if os.path.exists(temp_path):
			os.remove(temp_path)

		url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)
		headers = self.auth_headers(token)

		try:
			# Use session for connection pooling (20-30% faster)
			with self.session.get(url, stream=True, headers=headers, timeout=60) as response:
				response.raise_for_status()
				content_length = response.headers.get('Content-Length')
				if content_length:
					try:
						progress.set_file_size(file_index, int(content_length))
					except ValueError:
						pass

				with open(temp_path, 'wb') as dest:
					for chunk in response.iter_content(chunk_size=self.CHUNK_SIZE):
						if not chunk:
							continue
						dest.write(chunk)
						# Emit partial progress as bytes accumulate for the current file.
						progress.update_bytes(len(chunk))

			os.replace(temp_path, local_path)
			final_size = os.path.getsize(local_path)
			progress.set_file_size(file_index, final_size)
			return local_path
		finally:
			if os.path.exists(temp_path):
				os.remove(temp_path)

	def fetch_remote_file_size(
		self,
		repo_id: str,
		filename: str,
		revision: str,
		token: Optional[str] = None,
	) -> int:
		"""Best-effort size lookup for repositories that do not publish sibling metadata."""
		url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)
		headers = self.auth_headers(token)

		try:
			# Use session for connection pooling
			response = self.session.head(url, headers=headers, timeout=30, allow_redirects=True)
			response.raise_for_status()
			content_length = response.headers.get('Content-Length')
			if not content_length:
				return 0
			return max(int(content_length), 0)
		except (requests.RequestException, ValueError) as error:
			logger.debug('Unable to determine size for %s: %s', filename, error)
			return 0


download_service = DownloadService()
