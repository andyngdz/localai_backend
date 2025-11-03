import asyncio
import fnmatch
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import requests
from huggingface_hub import HfApi
from sqlalchemy.orm import Session

from app.services import logger_service
from app.services.models import model_service
from app.services.storage import storage_service

from .file_downloader import FileDownloader
from .filters import get_ignore_components
from .progress import DownloadProgress
from .repository import HuggingFaceRepository

# Backwards compatibility for tests stubbing the old symbol.
DownloadTqdm = DownloadProgress

logger = logger_service.get_logger(__name__, category='Download')


class DownloadService:
	"""Service responsible for downloading models and emitting socket progress."""

	def __init__(
		self,
		api: Optional[HfApi] = None,
		executor: Optional[ThreadPoolExecutor] = None,
		session: Optional[requests.Session] = None,
	):
		self.executor = executor or ThreadPoolExecutor()
		self.repository = HuggingFaceRepository(api=api)
		self.file_downloader = FileDownloader(session=session)

	async def start(self, id: str, db: Session):
		loop = asyncio.get_event_loop()
		local_dir = await loop.run_in_executor(self.executor, self.download_model, id, db)
		return local_dir

	def download_model(self, id: str, db: Session):
		"""Download a model from HuggingFace Hub with progress tracking."""
		if not id or not id.strip():
			raise ValueError('Model ID cannot be empty')

		repo_info = self.repository.get_repo_info(id)
		revision = getattr(repo_info, 'sha', 'main')
		# Build the list of candidate files and initial size map up-front so byte totals remain monotonic.
		components = self.repository.get_components(id, revision=revision)
		components_scopes = [f'{component}/*' for component in components]
		files = self.repository.list_files(id, repo_info=repo_info)
		ignore_components = get_ignore_components(files, components_scopes)
		file_sizes_map = self.repository.get_file_sizes_map(id, repo_info=repo_info)

		files_to_download = [
			file_path
			for file_path in files
			if (file_path == 'model_index.json' or any(fnmatch.fnmatch(file_path, scope) for scope in components_scopes))
			and file_path not in ignore_components
		]

		files_to_download.sort(key=lambda file_path: (file_path != 'model_index.json', file_path))
		# Ensure every file has deterministic size before streaming; fall back to HEAD if missing.
		file_sizes: List[int] = []
		for filename in files_to_download:
			size = file_sizes_map.get(filename, 0)
			if size <= 0:
				size = self.file_downloader.fetch_remote_file_size(id, filename, revision=revision)
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
				progress.start_file(filename)
				logger.info('Downloading %s (%s/%s)', filename, index + 1, total)
				local_path = self.file_downloader.download_file(
					repo_id=id,
					filename=filename,
					revision=revision,
					snapshot_dir=snapshot_dir,
					file_index=index,
					progress=progress,
					file_size=file_sizes[index],
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


download_service = DownloadService()
