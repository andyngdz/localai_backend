import importlib
import json
import os
import pathlib
import sys
from types import ModuleType, SimpleNamespace
from typing import List, Tuple
from unittest.mock import MagicMock

import pytest


def import_services_with_stubs(dummy_socket: object | None = None):
	"""Import app.features.downloads.services while stubbing heavy deps if missing.

	- Stubs huggingface_hub (HfApi, hf_hub_download)
	- Stubs tqdm.tqdm base class used for progress
	- Stubs app.socket to avoid importing python-socketio
	"""
	# Stub huggingface_hub if not installed
	if 'huggingface_hub' not in sys.modules:
		hf_mod = ModuleType('huggingface_hub')

		class HfApi:  # minimal placeholder, tests may overwrite service.api
			def repo_info(self, repo_id: str):  # pragma: no cover - replaced in tests
				return SimpleNamespace(siblings=[])

		def hf_hub_download(**_kwargs):  # pragma: no cover - replaced in tests
			raise RuntimeError('hf_hub_download should be stubbed in tests')

		hf_mod.HfApi = HfApi
		hf_mod.hf_hub_download = hf_hub_download
		sys.modules['huggingface_hub'] = hf_mod

	# Stub tqdm if not installed
	if 'tqdm' not in sys.modules:
		tqdm_mod = ModuleType('tqdm')

		class tqdm:
			def __init__(self, *args, **kwargs):
				self.n = 0
				self.total = kwargs.get('total', 0)
				self.desc = kwargs.get('desc', '')

			def update(self, n: int = 1):
				self.n += n

			def close(self):
				pass

		tqdm_mod.tqdm = tqdm
		sys.modules['tqdm'] = tqdm_mod

	# Stub app.socket submodule early to avoid importing python-socketio
	socket_mod = ModuleType('app.socket')
	socket_mod.socket_service = dummy_socket or SimpleNamespace(
		download_step_progress=lambda data: None,
		download_start=lambda data: None,
	)
	sys.modules['app.socket'] = socket_mod

	# Stub packages 'app.features' and 'app.features.downloads' to prevent executing their __init__
	# Set __path__ so that Python can locate the real 'services.py' under these packages
	project_root = pathlib.Path(__file__).resolve().parents[4]
	features_pkg = ModuleType('app.features')
	setattr(features_pkg, '__path__', [str(project_root / 'app' / 'features')])
	sys.modules['app.features'] = features_pkg

	downloads_pkg = ModuleType('app.features.downloads')
	setattr(downloads_pkg, '__path__', [str(project_root / 'app' / 'features' / 'downloads')])
	sys.modules['app.features.downloads'] = downloads_pkg

	# Stub app.features.downloads.schemas to avoid importing pydantic
	schemas_mod = ModuleType('app.features.downloads.schemas')

	class DownloadStepProgressResponse:
		def __init__(self, id: str, step: int, total: int) -> None:
			self.id = id
			self.step = step
			self.total = total

	schemas_mod.DownloadStepProgressResponse = DownloadStepProgressResponse
	sys.modules['app.features.downloads.schemas'] = schemas_mod

	# Stub app.database.crud to avoid importing SQLAlchemy
	crud_mod = ModuleType('app.database.crud')

	def add_model(db, id, path):
		pass

	crud_mod.add_model = add_model
	sys.modules['app.database.crud'] = crud_mod

	# Now import the target module
	services = importlib.import_module('app.features.downloads.services')
	# Ensure the module-level binding points to our dummy socket when provided
	if dummy_socket is not None:
		setattr(services, 'socket_service', dummy_socket)
		
	# Replace the DownloadTqdm class with our stub that properly handles desc
	class StubDownloadTqdm:
		def __init__(self, *args, **kwargs):
			self.id = kwargs.pop('id')
			self.n = 0
			self.total = kwargs.get('total', 0)
			self.desc = kwargs.get('desc', '')
			
		def update(self, n=1):
			self.n += n
			if dummy_socket is not None:
				dummy_socket.download_step_progress(
					schemas_mod.DownloadStepProgressResponse(
						id=self.id,
						step=self.n,
						total=self.total,
					)
				)
				
		def close(self):
			pass
			
	setattr(services, 'DownloadTqdm', StubDownloadTqdm)
	return services

@pytest.mark.asyncio
async def test_start_invokes_download_model_in_executor(
	monkeypatch: pytest.MonkeyPatch,
):
	# Arrange
	services = import_services_with_stubs()
	service = services.DownloadService()
	mock_db = MagicMock()

	# Arrange: Mock the synchronous download_model method
	mock_download_model = MagicMock(return_value='/fake/path')
	monkeypatch.setattr(service, 'download_model', mock_download_model)

	# Act
	result = await service.start('some/repo', mock_db)

	# Assert
	assert result == '/fake/path'
	mock_download_model.assert_called_once_with('some/repo', mock_db)


def test_download_model_handles_database_exception(
	tmp_path: pathlib.Path,
	monkeypatch: pytest.MonkeyPatch,
):
	# Arrange
	services = import_services_with_stubs()
	service = services.DownloadService()
	mock_db = MagicMock()

	# Arrange: Mock file discovery and download
	monkeypatch.setattr(service, 'get_components', lambda _id: ['unet'])
	monkeypatch.setattr(service, 'list_files', lambda _id: ['unet/model.bin'])
	snapshot_root = tmp_path / 'snap-123'
	def fake_hf_hub_download(**_kwargs):
		local_path = snapshot_root / _kwargs['filename']
		local_path.parent.mkdir(parents=True, exist_ok=True)
		local_path.write_bytes(b'')
		return str(local_path)
	monkeypatch.setattr(services, 'hf_hub_download', fake_hf_hub_download)

	# Arrange: Mock add_model to raise an error
	def fake_add_model(db, id, path):
		raise ValueError('DB error')
	monkeypatch.setattr(services, 'add_model', fake_add_model)

	# Arrange: Spy on the logger
	mock_logger = MagicMock()
	monkeypatch.setattr(services, 'logger', mock_logger)

	# Act
	local_dir = service.download_model('some/repo', mock_db)

	# Assert
	assert local_dir == os.path.join(str(snapshot_root), 'unet')
	mock_logger.error.assert_called_once_with(
		"Failed to save model some/repo to database: DB error"
	)


def test_download_model_handles_download_exception(
	tmp_path: pathlib.Path,
	monkeypatch: pytest.MonkeyPatch,
):
	# Arrange
	services = import_services_with_stubs()
	service = services.DownloadService()
	mock_db = MagicMock()

	# Arrange: Mock file discovery
	monkeypatch.setattr(service, 'get_components', lambda _id: ['unet'])
	monkeypatch.setattr(service, 'list_files', lambda _id: ['unet/model.bin'])

	# Arrange: Mock hf_hub_download to raise an error
	def fake_hf_hub_download(**_kwargs):
		raise ConnectionError('Download failed')

	monkeypatch.setattr(services, 'hf_hub_download', fake_hf_hub_download)

	# Arrange: Spy on the progress bar's close method
	mock_progress_close = MagicMock()

	class MockDownloadTqdm:
		def __init__(self, *args, **kwargs):
			pass

		def update(self, n=1):
			pass

		def close(self):
			mock_progress_close()

	monkeypatch.setattr(services, 'DownloadTqdm', MockDownloadTqdm)

	# Act & Assert
	with pytest.raises(ConnectionError, match='Download failed'):
		service.download_model('some/repo', mock_db)

	mock_progress_close.assert_called_once()


def test_download_model_when_no_files_to_download(
	monkeypatch: pytest.MonkeyPatch,
):
	# Arrange
	services = import_services_with_stubs()
	service = services.DownloadService()
	mock_db = MagicMock()

	# Arrange: Mock dependencies to return empty file list
	monkeypatch.setattr(service, 'get_components', lambda _id: [])
	monkeypatch.setattr(service, 'list_files', lambda _id: [])
	
	# Arrange: Spy on the logger
	mock_logger = MagicMock()
	monkeypatch.setattr(services, 'logger', mock_logger)

	# Act
	result = service.download_model('some/repo', mock_db)

	# Assert
	assert result is None
	mock_logger.warning.assert_called_once_with('No files to download')


class DummySocket:
	def __init__(self) -> None:
		self.progress_calls: List[Tuple[str, int, int]] = []

	def download_step_progress(self, data):  # matches BaseModel-like interface
		self.progress_calls.append((data.id, data.step, data.total))


def test_get_ignore_components_filters_bin_when_safetensors_present():
	# Arrange
	services = import_services_with_stubs()
	service = services.DownloadService()
	files = [
		'unet/model.bin',
		'unet/model.safetensors',
		'vae/model.bin',
		'text_encoder/model.safetensors',
	]
	scopes = ['unet/*', 'vae/*']

	# Act
	ignored = service.get_ignore_components(files, scopes)

	# Assert
	assert ignored == ['unet/model.bin']


def test_get_components_parses_model_index_json(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
	# Arrange
	services = import_services_with_stubs()
	model_index = {
		'unet': ['some-config'],
		'vae': [None],  # should be ignored because first element is None
		'scheduler': 'not-a-list',  # ignored
	}
	model_index_path = tmp_path / 'model_index.json'
	model_index_path.write_text(json.dumps(model_index), encoding='utf-8')

	def fake_hf_hub_download(
		*,
		repo_id: str,
		filename: str,
		repo_type: str | None = None,
		cache_dir: str | None = None,
	):
		assert filename == 'model_index.json'
		return str(model_index_path)

	monkeypatch.setattr(services, 'hf_hub_download', fake_hf_hub_download)
	service = services.DownloadService()

	# Act
	components = service.get_components('some/repo')

	# Assert
	assert components == ['unet']


def test_list_files_returns_siblings(monkeypatch: pytest.MonkeyPatch):
	# Arrange
	services = import_services_with_stubs()
	service = services.DownloadService()

	siblings = [SimpleNamespace(rfilename='a.txt'), SimpleNamespace(rfilename='b/c.bin')]

	class DummyApi:
		def repo_info(self, repo_id: str):
			assert repo_id == 'some/repo'
			return SimpleNamespace(siblings=siblings)

	service.api = DummyApi()  # type: ignore[assignment]

	# Act
	result = service.list_files('some/repo')

	# Assert
	assert result == ['a.txt', 'b/c.bin']


def test_download_model_downloads_expected_files_and_returns_dir(
	tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
	# Arrange
	dummy_socket = DummySocket()
	services = import_services_with_stubs(dummy_socket)
	service = services.DownloadService()

	# Create mock db session
	mock_db = MagicMock()

	# Components that we care about
	monkeypatch.setattr(service, 'get_components', lambda _id: ['unet', 'vae'])  # type: ignore[misc]

	# File listing contains one safetensors/bin pair under unet, and only bin under vae
	def fake_list_files(_id: str) -> List[str]:
		return [
			'README.md',  # should be ignored (not in components scopes)
			'model_index.json',  # always included
			'unet/model.safetensors',  # included
			'unet/model.bin',  # ignored due to safetensors
			'vae/model.bin',  # included
		]

	monkeypatch.setattr(service, 'list_files', fake_list_files)  # type: ignore[misc]

	# Fake download to create paths under a snapshot directory
	calls: List[Tuple[str, str]] = []
	snapshot_root = tmp_path / 'snap-123'

	def fake_hf_hub_download(*, repo_id: str, filename: str, cache_dir: str | None = None):
		calls.append((repo_id, filename))
		local_path = snapshot_root / filename
		local_path.parent.mkdir(parents=True, exist_ok=True)
		# create empty file to simulate presence
		local_path.write_bytes(b'')
		return str(local_path)

	monkeypatch.setattr(services, 'hf_hub_download', fake_hf_hub_download)

	# Mock add_model function to verify it's called
	mock_add_model = MagicMock()
	monkeypatch.setattr(services, 'add_model', mock_add_model)

	# Act
	local_dir = service.download_model('some/repo', mock_db)

	# Assert: files downloaded in correct order with model_index.json first
	expected_filenames = ['model_index.json', 'unet/model.safetensors', 'vae/model.bin']
	assert [f for (_repo, f) in calls] == expected_filenames

	# Assert: returned snapshot directory is the parent of first downloaded file
	assert os.path.normpath(local_dir) == os.path.normpath(str(snapshot_root))

	# Assert: progress events emitted with correct steps and totals
	assert [step for (_id, step, _total) in dummy_socket.progress_calls] == [1, 2, 3]
	assert all(total == 3 for (_id, _step, total) in dummy_socket.progress_calls)
	assert all(_id == 'some/repo' for (_id, _step, _total) in dummy_socket.progress_calls)

	# Assert: add_model was called with the correct parameters
	mock_add_model.assert_called_once_with(mock_db, 'some/repo', os.path.normpath(str(snapshot_root)))
