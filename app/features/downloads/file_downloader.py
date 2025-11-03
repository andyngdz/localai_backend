import os
from pathlib import Path
from typing import Dict, Optional

import requests
from huggingface_hub import hf_hub_url
from requests.adapters import HTTPAdapter

from app.services import logger_service

from .progress import DownloadProgress

logger = logger_service.get_logger(__name__, category='Download')


class FileDownloader:
	"""Handles low-level file download operations from HuggingFace Hub."""

	CHUNK_SIZE = 4 * 1024 * 1024

	def __init__(self, session: Optional[requests.Session] = None):
		self.session = session or self._build_session()

	def _build_session(self) -> requests.Session:
		session = requests.Session()
		adapter = HTTPAdapter(
			pool_connections=10,
			pool_maxsize=20,
			max_retries=3,
		)
		session.mount('https://', adapter)
		return session

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
		progress: DownloadProgress,
		file_size: Optional[int] = None,
		token: Optional[str] = None,
	) -> str:
		"""
		Download a single file from HuggingFace Hub with streaming and progress tracking.

		Args:
			repo_id: HuggingFace repository ID
			filename: Relative path of file within repository (validated for path traversal)
			revision: Git revision/commit hash
			snapshot_dir: Local directory to store downloaded files
			file_index: Zero-based index in progress.file_sizes list (must align with file order)
			progress: Progress tracker for websocket updates
			file_size: Expected size for the file (used to seed progress totals)
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
		snapshot_path = Path(snapshot_dir)
		local_path = snapshot_path / filename

		os.makedirs(snapshot_path, exist_ok=True)
		os.makedirs(local_path.parent, exist_ok=True)

		local_path_str = str(local_path)

		if os.path.exists(local_path_str):
			actual_size = os.path.getsize(local_path_str)
			if actual_size > 0:
				progress.set_file_size(file_index, actual_size)
				logger.debug('Skipping download for %s; already complete', filename)
				return local_path_str
			os.remove(local_path_str)

		temp_path = f'{local_path_str}.part'
		if os.path.exists(temp_path):
			os.remove(temp_path)

		url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)
		headers = self.auth_headers(token)

		try:
			with self.session.get(url, stream=True, headers=headers, timeout=60) as response:
				response.raise_for_status()
				target_size = file_size
				content_length = response.headers.get('Content-Length')
				if content_length:
					try:
						target_size = int(content_length)
					except (TypeError, ValueError):
						target_size = file_size
				if target_size and target_size > 0:
					progress.set_file_size(file_index, target_size)

				with open(temp_path, 'wb') as dest:
					for chunk in response.iter_content(chunk_size=self.CHUNK_SIZE):
						if not chunk:
							continue
						dest.write(chunk)
						progress.update_bytes(len(chunk))

			os.replace(temp_path, local_path_str)
			final_size = os.path.getsize(local_path_str)
			progress.set_file_size(file_index, final_size)
			return local_path_str
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
			response = self.session.head(url, headers=headers, timeout=30, allow_redirects=True)
			response.raise_for_status()
			content_length = response.headers.get('Content-Length')
			if not content_length:
				return 0
			return max(int(content_length), 0)
		except (requests.RequestException, ValueError) as error:
			logger.debug('Unable to determine size for %s: %s', filename, error)
			return 0
