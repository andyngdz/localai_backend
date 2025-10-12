import importlib
import json
import os
import pathlib
import sys
from types import ModuleType, SimpleNamespace
from typing import List, Tuple
from unittest.mock import MagicMock

import pytest


class DummySocket:
	def __init__(self) -> None:
		self.progress_calls: List[Tuple[str, int, int, int, int, str, str | None]] = []

	# Make this a properly typed method
	def download_step_progress(self, data) -> None:  # matches BaseModel-like interface
		self.progress_calls.append(
			(
				data.id,
				data.step,
				data.total,
				data.downloaded_size,
				data.total_downloaded_size,
				data.phase,
				getattr(data, 'current_file', None),
			)
		)

	def download_start(self, data) -> None:
		pass


def import_services_with_stubs(dummy_socket: 'DummySocket | None' = None):
	"""Import app.features.downloads.services while stubbing heavy deps if missing.

	- Stubs huggingface_hub (HfApi, hf_hub_download)
	- Stubs tqdm.tqdm base class used for progress
	- Stubs app.socket to avoid importing python-socketio
	"""
	# Stub huggingface_hub if not installed
	if 'huggingface_hub' not in sys.modules:
		hf_mod = ModuleType('huggingface_hub')
		sys.modules['huggingface_hub'] = hf_mod

		class HfApi:  # minimal placeholder, tests may overwrite service.api
			def repo_info(self, repo_id: str):  # pragma: no cover - replaced in tests
				return SimpleNamespace(siblings=[])

		def hf_hub_download(**_kwargs):  # pragma: no cover - replaced in tests
			raise RuntimeError('hf_hub_download should be stubbed in tests')

		def hf_hub_url(*, repo_id: str, filename: str, revision: str | None = None):
			return f'https://example.com/{repo_id}/{revision or "main"}/{filename}'

		# Use setattr for type-safe attribute assignment
		setattr(hf_mod, 'HfApi', HfApi)
		setattr(hf_mod, 'hf_hub_download', hf_hub_download)
		setattr(hf_mod, 'hf_hub_url', hf_hub_url)

	# Stub tqdm if not installed
	if 'tqdm' not in sys.modules:
		tqdm_mod = ModuleType('tqdm')
		sys.modules['tqdm'] = tqdm_mod

		class tqdm:
			def __init__(self, *args, **kwargs):
				self.n = 0
				self.total = kwargs.get('total', 0)
				self.desc = kwargs.get('desc', '')

			def update(self, n: int = 1):
				self.n += n

			def close(self):
				pass

		# Use setattr for type-safe attribute assignment
		setattr(tqdm_mod, 'tqdm', tqdm)

	# Stub app.services.styles to avoid importing transformers
	styles_mod = ModuleType('app.services.styles')
	sys.modules['app.services.styles'] = styles_mod
	
	class StylesService:
		def __init__(self):
			pass
			
	styles_service = StylesService()
	setattr(styles_mod, 'styles_service', styles_service)
	
	# Stub app.services.models to avoid importing real dependencies
	models_mod = ModuleType('app.services.models')
	sys.modules['app.services.models'] = models_mod
	
	class ModelService:
		def add_model(self, db, id, path):
			pass
			
	model_service = ModelService()
	setattr(models_mod, 'model_service', model_service)

	# Stub app.services.storage so we don't rely on filesystem layout
	storage_mod = ModuleType('app.services.storage')
	sys.modules['app.services.storage'] = storage_mod

	class StorageService:
		def get_model_dir(self, id: str) -> str:
			return os.path.join('/tmp/localai-tests', id.replace('/', '--'))

		def get_model_lock_dir(self, id: str) -> str:
			return os.path.join('/tmp/localai-tests', 'locks', id.replace('/', '--'))

	storage_service = StorageService()
	setattr(storage_mod, 'storage_service', storage_service)

	# Stub app.socket submodule early to avoid importing python-socketio
	socket_mod = ModuleType('app.socket')
	sys.modules['app.socket'] = socket_mod

	# Use setattr for type-safe attribute assignment
	setattr(
		socket_mod,
		'socket_service',
		dummy_socket
		or SimpleNamespace(
			download_step_progress=lambda data: None,
			download_start=lambda data: None,
		),
	)

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
	sys.modules['app.features.downloads.schemas'] = schemas_mod

	class DownloadStepProgressResponse:
		def __init__(
			self,
			id: str,
			step: int,
			total: int,
			downloaded_size: int,
			total_downloaded_size: int,
			phase: str,
			current_file: str | None = None,
		) -> None:
			self.id = id
			self.step = step
			self.total = total
			self.downloaded_size = downloaded_size
			self.total_downloaded_size = total_downloaded_size
			self.phase = phase
			self.current_file = current_file

	# Use setattr for type-safe attribute assignment
	setattr(schemas_mod, 'DownloadStepProgressResponse', DownloadStepProgressResponse)

	# Stub app.database.crud to avoid importing SQLAlchemy
	crud_mod = ModuleType('app.database.crud')
	sys.modules['app.database.crud'] = crud_mod

	def add_model(db, id, path):
		pass

	# Use setattr for type-safe attribute assignment
	setattr(crud_mod, 'add_model', add_model)

	# Now import the target module
	services = importlib.import_module('app.features.downloads.services')
	# Ensure the module-level binding points to our dummy socket when provided
	if dummy_socket is not None:
		setattr(services, 'socket_service', dummy_socket)

	# Replace the DownloadTqdm class with our stub that properly handles desc
	class StubDownloadTqdm:
		def __init__(self, *args, **kwargs):
			self.id = kwargs.pop('id')
			self.file_sizes = kwargs.pop('file_sizes', [])
			self.n = 0
			self.total = kwargs.get('total', 0)
			self.desc = kwargs.get('desc', '')
			self.downloaded_size = 0
			self.total_downloaded_size = sum(self.file_sizes)
			self.current_file: str | None = None
			self.emit_progress('init')

		def emit_progress(self, phase: str):
			if dummy_socket is not None:
				response = getattr(sys.modules['app.features.downloads.schemas'], 'DownloadStepProgressResponse')(
					id=self.id,
					step=self.n,
					total=self.total,
					downloaded_size=self.downloaded_size,
					total_downloaded_size=self.total_downloaded_size,
					phase=phase,
					current_file=self.current_file,
				)
				dummy_socket.download_step_progress(response)

		def start_file(self, filename: str):
			self.current_file = filename
			self.emit_progress('file_start')

		def update(self, n=1):
			self.n += n
			if self.n > 0:
				completed = sum(self.file_sizes[: self.n])
				if self.downloaded_size < completed:
					self.downloaded_size = completed
			self.emit_progress('file_complete')

		def set_file_size(self, index: int, size: int):
			if index < 0:
				return
			if index >= len(self.file_sizes):
				self.file_sizes.extend([0] * (index + 1 - len(self.file_sizes)))
			previous = self.file_sizes[index]
			if previous == size:
				return
			self.file_sizes[index] = size
			self.total_downloaded_size += size - previous
			if self.total_downloaded_size < 0:
				self.total_downloaded_size = 0
			self.emit_progress('size_update')

		def update_bytes(self, byte_count: int):
			if byte_count <= 0:
				return
			self.downloaded_size += byte_count
			if self.total_downloaded_size and self.downloaded_size > self.total_downloaded_size:
				self.downloaded_size = self.total_downloaded_size
			self.emit_progress('chunk')

		def close(self):
			pass

	setattr(services, 'DownloadTqdm', StubDownloadTqdm)
	setattr(services.DownloadService, 'fetch_remote_file_size', lambda self, repo_id, filename, revision, token=None: 0)
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
	monkeypatch.setattr(service, 'get_components', lambda _id, revision=None: ['unet'])
	monkeypatch.setattr(service, 'list_files', lambda _id, repo_info=None: ['unet/model.bin'])
	model_root = tmp_path / 'cache-db'
	monkeypatch.setattr(services.storage_service, 'get_model_dir', lambda _id: str(model_root))

	model_index_path = tmp_path / 'model_index.json'
	model_index_path.write_text(json.dumps({'unet': ['cfg']}), encoding='utf-8')

	def fake_hf_hub_download(**_kwargs):
		return str(model_index_path)

	monkeypatch.setattr(services, 'hf_hub_download', fake_hf_hub_download)

	class DummyApi:
		def repo_info(self, repo_id: str):
			return SimpleNamespace(
				sha='main',
				siblings=[
					SimpleNamespace(rfilename='model_index.json', size=10),
					SimpleNamespace(rfilename='unet/model.bin', size=25),
				],
			)

	service.api = DummyApi()  # type: ignore[assignment]

	def fake_download_file(
		self,
		repo_id: str,
		filename: str,
		revision: str,
		snapshot_dir: str,
		file_index: int,
		file_size: int,
		progress,
		token=None,
	):
		local_path = pathlib.Path(snapshot_dir) / filename
		local_path.parent.mkdir(parents=True, exist_ok=True)
		size = file_size or 5
		with open(local_path, 'wb') as dest:
			remaining = size
			while remaining > 0:
				chunk = min(remaining, max(1, size // 2))
				dest.write(b'x' * chunk)
				progress.update_bytes(chunk)
				remaining -= chunk
		return str(local_path)

	monkeypatch.setattr(services.DownloadService, 'download_file', fake_download_file, raising=False)

	# Arrange: Mock add_model to raise an error
	def fake_add_model(db, id, path):
		raise ValueError('DB error')

	monkeypatch.setattr(services.model_service, 'add_model', fake_add_model)

	# Arrange: Spy on the logger
	mock_logger = MagicMock()
	monkeypatch.setattr(services, 'logger', mock_logger)

	# Act
	local_dir = service.download_model('some/repo', mock_db)

	# Assert
	expected_snapshot = os.path.join(str(model_root), 'snapshots', 'main', 'unet')
	assert os.path.normpath(local_dir) == os.path.normpath(expected_snapshot)
	mock_logger.error.assert_called_once_with('Failed to save model some/repo to database: DB error')


def test_download_model_handles_download_exception(
	tmp_path: pathlib.Path,
	monkeypatch: pytest.MonkeyPatch,
):
	# Arrange
	services = import_services_with_stubs()
	service = services.DownloadService()
	mock_db = MagicMock()

	# Arrange: Mock file discovery
	monkeypatch.setattr(service, 'get_components', lambda _id, revision=None: ['unet'])
	monkeypatch.setattr(service, 'list_files', lambda _id, repo_info=None: ['unet/model.bin'])
	monkeypatch.setattr(
		service,
		'get_file_sizes_map',
		lambda _id, repo_info=None: {'unet/model.bin': 25},
	)

	model_root = tmp_path / 'cache-failure'
	monkeypatch.setattr(services.storage_service, 'get_model_dir', lambda _id: str(model_root))

	# Arrange: Mock hf_hub_download to raise an error
	model_index_path = tmp_path / 'model_index.json'
	model_index_path.write_text(json.dumps({'unet': ['cfg']}), encoding='utf-8')

	def fake_hf_hub_download(**_kwargs):
		return str(model_index_path)

	monkeypatch.setattr(services, 'hf_hub_download', fake_hf_hub_download)

	class DummyApi:
		def repo_info(self, repo_id: str):
			return SimpleNamespace(sha='main', siblings=[])

	service.api = DummyApi()  # type: ignore[assignment]

	def fake_download_file(
		self,
		repo_id: str,
		filename: str,
		revision: str,
		snapshot_dir: str,
		file_index: int,
		file_size: int,
		progress,
		token=None,
	):
		raise ConnectionError('Download failed')

	monkeypatch.setattr(services.DownloadService, 'download_file', fake_download_file, raising=False)

	# Arrange: Spy on the progress bar's close method
	mock_progress_close = MagicMock()

	class MockDownloadTqdm:
		def __init__(self, *args, **kwargs):
			pass

		def start_file(self, filename):
			pass

		def update(self, n=1):
			pass

		def set_file_size(self, index, size):
			pass

		def update_bytes(self, byte_count):
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

	class DummyApi:
		def repo_info(self, repo_id: str):
			return SimpleNamespace(sha='main', siblings=[])

	service.api = DummyApi()  # type: ignore[assignment]

	# Arrange: Mock dependencies to return empty file list
	monkeypatch.setattr(service, 'get_components', lambda _id, revision=None: [])
	monkeypatch.setattr(service, 'list_files', lambda _id, repo_info=None: [])

	# Arrange: Spy on the logger
	mock_logger = MagicMock()
	monkeypatch.setattr(services, 'logger', mock_logger)

	# Act
	result = service.download_model('some/repo', mock_db)

	# Assert
	assert result is None
	mock_logger.warning.assert_called_once_with('No files to download')


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


def test_get_ignore_components_keeps_bin_when_no_safetensors():
	# Arrange
	services = import_services_with_stubs()
	service = services.DownloadService()
	files = [
		'unet/model.bin',
		'vae/model.bin',
		'scheduler/config.json',
	]
	scopes = ['unet/*', 'vae/*']

	# Act
	ignored = service.get_ignore_components(files, scopes)

	# Assert
	assert ignored == []


def test_get_ignore_components_handles_empty_files():
	# Arrange
	services = import_services_with_stubs()
	service = services.DownloadService()
	files = []
	scopes = ['unet/*', 'vae/*']

	# Act
	ignored = service.get_ignore_components(files, scopes)

	# Assert
	assert ignored == []


def test_get_ignore_components_handles_out_of_scope_files():
	# Arrange
	services = import_services_with_stubs()
	service = services.DownloadService()
	files = [
		'README.md',
		'config.json',
		'out_of_scope/model.bin',
		'out_of_scope/model.safetensors',
	]
	scopes = ['unet/*', 'vae/*']

	# Act
	ignored = service.get_ignore_components(files, scopes)

	# Assert
	assert ignored == []


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
		revision: str | None = None,
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


def test_get_components_handles_malformed_json(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
	# Arrange
	services = import_services_with_stubs()
	model_index_path = tmp_path / 'model_index.json'
	model_index_path.write_text('invalid json', encoding='utf-8')

	def fake_hf_hub_download(
		*,
		repo_id: str,
		filename: str,
		repo_type: str | None = None,
		revision: str | None = None,
		cache_dir: str | None = None,
	):
		return str(model_index_path)

	monkeypatch.setattr(services, 'hf_hub_download', fake_hf_hub_download)
	service = services.DownloadService()

	# Act & Assert
	with pytest.raises(json.JSONDecodeError):
		service.get_components('some/repo')


def test_get_components_handles_empty_dict(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
	# Arrange
	services = import_services_with_stubs()
	model_index = {}
	model_index_path = tmp_path / 'model_index.json'
	model_index_path.write_text(json.dumps(model_index), encoding='utf-8')

	def fake_hf_hub_download(
		*,
		repo_id: str,
		filename: str,
		repo_type: str | None = None,
		revision: str | None = None,
		cache_dir: str | None = None,
	):
		return str(model_index_path)

	monkeypatch.setattr(services, 'hf_hub_download', fake_hf_hub_download)
	service = services.DownloadService()

	# Act
	components = service.get_components('some/repo')

	# Assert
	assert components == []


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
	monkeypatch.setattr(service, 'get_components', lambda _id, revision=None: ['unet', 'vae'])  # type: ignore[misc]

	# File listing contains one safetensors/bin pair under unet, and only bin under vae
	def fake_list_files(_id: str, repo_info=None) -> List[str]:
		return [
			'README.md',  # should be ignored (not in components scopes)
			'model_index.json',  # always included
			'unet/model.safetensors',  # included
			'unet/model.bin',  # ignored due to safetensors
			'vae/model.bin',  # included
		]

	monkeypatch.setattr(service, 'list_files', fake_list_files)  # type: ignore[misc]
	monkeypatch.setattr(
		service,
		'get_file_sizes_map',
		lambda _id, repo_info=None: {
			'model_index.json': 10,
			'unet/model.safetensors': 20,
			'vae/model.bin': 30,
		},
	)

	class DummyApi:
		def repo_info(self, repo_id: str):
			return SimpleNamespace(
				sha='main',
				siblings=[
					SimpleNamespace(rfilename='model_index.json', size=10),
					SimpleNamespace(rfilename='unet/model.safetensors', size=20),
					SimpleNamespace(rfilename='vae/model.bin', size=30),
				],
			)

	service.api = DummyApi()  # type: ignore[assignment]

	# Fake download to create paths under a snapshot directory
	model_root = tmp_path / 'cache-root'
	monkeypatch.setattr(services.storage_service, 'get_model_dir', lambda _id: str(model_root))

	model_index_path = tmp_path / 'model_index.json'
	model_index_path.write_text(json.dumps({'unet': ['cfg'], 'vae': ['cfg']}), encoding='utf-8')

	def fake_hf_hub_download(*, repo_id: str, filename: str, **_kwargs):
		assert repo_id == 'some/repo'
		assert filename == 'model_index.json'
		return str(model_index_path)

	monkeypatch.setattr(services, 'hf_hub_download', fake_hf_hub_download)

	download_calls: List[str] = []

	def fake_download_file(
		self,
		repo_id: str,
		filename: str,
		revision: str,
		snapshot_dir: str,
		file_index: int,
		file_size: int,
		progress,
		token=None,
	):
		assert repo_id == 'some/repo'
		download_calls.append(filename)
		local_path = pathlib.Path(snapshot_dir) / filename
		local_path.parent.mkdir(parents=True, exist_ok=True)
		size = file_size or 5
		with open(local_path, 'wb') as dest:
			remaining = size
			chunk = max(1, size // 2)
			while remaining > 0:
				current = min(chunk, remaining)
				dest.write(b'x' * current)
				progress.update_bytes(current)
				remaining -= current
		return str(local_path)

	monkeypatch.setattr(services.DownloadService, 'download_file', fake_download_file, raising=False)

	# Mock model_service.add_model function to verify it's called
	mock_add_model = MagicMock()
	monkeypatch.setattr(services.model_service, 'add_model', mock_add_model)

	# Act
	local_dir = service.download_model('some/repo', mock_db)

	# Assert: files downloaded in correct order with model_index.json first
	expected_filenames = ['model_index.json', 'unet/model.safetensors', 'vae/model.bin']
	assert download_calls == expected_filenames

	# Assert: returned snapshot directory is the parent of first downloaded file
	expected_snapshot = os.path.join(str(model_root), 'snapshots', 'main')
	assert os.path.normpath(local_dir) == os.path.normpath(expected_snapshot)

	# Assert: progress events include chunk updates and final step totals
	completions = [call for call in dummy_socket.progress_calls if call[5] == 'file_complete']
	assert [call[1] for call in completions] == [1, 2, 3]
	assert completions[0][3] == 10
	assert completions[1][3] == 30
	assert completions[2][3] == 60
	assert completions[2][4] == 60

	# Assert: add_model was called with the correct parameters
	mock_add_model.assert_called_once_with(mock_db, 'some/repo', os.path.normpath(expected_snapshot))


def test_download_model_fetches_remote_sizes_when_missing(
	tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
	# Arrange
	dummy_socket = DummySocket()
	services = import_services_with_stubs(dummy_socket)
	service = services.DownloadService()
	mock_db = MagicMock()

	monkeypatch.setattr(service, 'get_components', lambda _id, revision=None: ['unet'])  # type: ignore[misc]

	monkeypatch.setattr(service, 'list_files', lambda _id, repo_info=None: ['model_index.json', 'unet/model.bin'])  # type: ignore[misc]
	monkeypatch.setattr(service, 'get_file_sizes_map', lambda _id, repo_info=None: {
		'model_index.json': 0,
		'unet/model.bin': 0,
	})

	class DummyApi:
		def repo_info(self, repo_id: str):
			return SimpleNamespace(
				sha='main',
				siblings=[
					SimpleNamespace(rfilename='model_index.json', size=0),
					SimpleNamespace(rfilename='unet/model.bin', size=0),
				],
			)

	service.api = DummyApi()  # type: ignore[assignment]

	model_root = tmp_path / 'cache-remote'
	monkeypatch.setattr(services.storage_service, 'get_model_dir', lambda _id: str(model_root))

	model_index_path = tmp_path / 'model_index.json'
	model_index_path.write_text(json.dumps({'unet': ['cfg']}), encoding='utf-8')

	def fake_hf_hub_download(*, repo_id: str, filename: str, **_kwargs):
		return str(model_index_path)

	monkeypatch.setattr(services, 'hf_hub_download', fake_hf_hub_download)

	fetch_calls: List[str] = []

	def fake_fetch_size(repo_id: str, filename: str, revision: str, token=None):
		fetch_calls.append(filename)
		return {'model_index.json': 12, 'unet/model.bin': 34}[filename]

	monkeypatch.setattr(service, 'fetch_remote_file_size', fake_fetch_size)

	def fake_download_file(
		self,
		repo_id: str,
		filename: str,
		revision: str,
		snapshot_dir: str,
		file_index: int,
		file_size: int,
		progress,
		token=None,
	):
		assert file_size == {'model_index.json': 12, 'unet/model.bin': 34}[filename]
		local_path = pathlib.Path(snapshot_dir) / filename
		local_path.parent.mkdir(parents=True, exist_ok=True)
		with open(local_path, 'wb') as dest:
			to_write = file_size
			while to_write > 0:
				chunk = min(10, to_write)
				dest.write(b'x' * chunk)
				progress.update_bytes(chunk)
				to_write -= chunk
		return str(local_path)

	monkeypatch.setattr(services.DownloadService, 'download_file', fake_download_file, raising=False)

	mock_add_model = MagicMock()
	monkeypatch.setattr(services.model_service, 'add_model', mock_add_model)

	# Act
	local_dir = service.download_model('some/repo', mock_db)

	# Assert
	assert fetch_calls == ['model_index.json', 'unet/model.bin']
	expected_snapshot = os.path.join(str(model_root), 'snapshots', 'main')
	assert os.path.normpath(local_dir) == os.path.normpath(expected_snapshot)

def test_download_tqdm_update_emits_progress():
	# Arrange
	dummy_socket = DummySocket()
	services = import_services_with_stubs(dummy_socket)
	
	# Use the actual DownloadTqdm class to test its behavior
	tqdm_instance = services.DownloadTqdm(
		id='test-repo',
		total=5,
		desc='Testing progress',
		file_sizes=[1, 1, 1, 1, 1],
	)
	
	# Act
	tqdm_instance.start_file('chunk.bin')
	tqdm_instance.update_bytes(2)
	tqdm_instance.update(1)

	# Assert
	assert len(dummy_socket.progress_calls) == 4
	init_call = dummy_socket.progress_calls[0]
	assert init_call[:5] == ('test-repo', 0, 5, 0, 5)
	assert init_call[5] == 'init'
	start_call = dummy_socket.progress_calls[1]
	assert start_call[5] == 'file_start'
	assert start_call[6] == 'chunk.bin'
	chunk_call = dummy_socket.progress_calls[2]
	assert chunk_call[5] == 'chunk'
	assert chunk_call[3] == 2
	complete_call = dummy_socket.progress_calls[3]
	assert complete_call[5] == 'file_complete'


def test_download_model_sorts_files_with_model_index_first(
	tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
	# Arrange
	services = import_services_with_stubs()
	service = services.DownloadService()
	mock_db = MagicMock()
	
	monkeypatch.setattr(service, 'get_components', lambda _id, revision=None: ['unet'])
	monkeypatch.setattr(service, 'list_files', lambda _id, repo_info=None: [
		'unet/config.json',
		'model_index.json',
		'unet/model.bin',
	])
	monkeypatch.setattr(
		service,
		'get_file_sizes_map',
		lambda _id, repo_info=None: {
			'model_index.json': 8,
			'unet/config.json': 12,
			'unet/model.bin': 16,
		},
	)

	class DummyApi:
		def repo_info(self, repo_id: str):
			return SimpleNamespace(sha='main', siblings=[])

	service.api = DummyApi()  # type: ignore[assignment]
	
	model_root = tmp_path / 'cache-sort'
	monkeypatch.setattr(services.storage_service, 'get_model_dir', lambda _id: str(model_root))
	
	model_index_path = tmp_path / 'model_index.json'
	model_index_path.write_text(json.dumps({'unet': ['cfg']}), encoding='utf-8')
	
	def fake_hf_hub_download(*, repo_id: str, filename: str, **_kwargs):
		return str(model_index_path)
	
	monkeypatch.setattr(services, 'hf_hub_download', fake_hf_hub_download)
	
	downloaded_files = []
	
	def fake_download_file(
		self,
		repo_id: str,
		filename: str,
		revision: str,
		snapshot_dir: str,
		file_index: int,
		file_size: int,
		progress,
		token=None,
	):
		downloaded_files.append(filename)
		local_path = pathlib.Path(snapshot_dir) / filename
		local_path.parent.mkdir(parents=True, exist_ok=True)
		with open(local_path, 'wb') as dest:
			dest.write(b'x' * max(file_size, 1))
			progress.update_bytes(max(file_size, 1))
		return str(local_path)
	
	monkeypatch.setattr(services.DownloadService, 'download_file', fake_download_file, raising=False)
	
	# Act
	service.download_model('test/repo', mock_db)
	
	# Assert: model_index.json should be downloaded first
	assert downloaded_files[0] == 'model_index.json'
	assert len(downloaded_files) == 3


def test_download_model_handles_file_filtering_logic(
	tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
	# Arrange
	services = import_services_with_stubs()
	service = services.DownloadService()
	mock_db = MagicMock()
	
	# Mock components and files to test filtering
	monkeypatch.setattr(service, 'get_components', lambda _id, revision=None: ['unet', 'vae'])
	monkeypatch.setattr(service, 'list_files', lambda _id, repo_info=None: [
		'README.md',  # should be filtered out
		'model_index.json',  # always included
		'unet/model.bin',
		'unet/model.safetensors',  # should cause unet/model.bin to be ignored
		'vae/config.json',
		'scheduler/model.bin',  # should be filtered out (not in components)
	])
	monkeypatch.setattr(
		service,
		'get_file_sizes_map',
		lambda _id, repo_info=None: {
			'model_index.json': 5,
			'unet/model.safetensors': 15,
			'vae/config.json': 7,
		},
	)

	class DummyApi:
		def repo_info(self, repo_id: str):
			return SimpleNamespace(sha='main', siblings=[])

	service.api = DummyApi()  # type: ignore[assignment]
	
	model_root = tmp_path / 'cache-filter'
	monkeypatch.setattr(services.storage_service, 'get_model_dir', lambda _id: str(model_root))
	
	model_index_path = tmp_path / 'model_index.json'
	model_index_path.write_text(json.dumps({'unet': ['cfg'], 'vae': ['cfg']}), encoding='utf-8')
	
	def fake_hf_hub_download(*, repo_id: str, filename: str, **_kwargs):
		return str(model_index_path)
	
	monkeypatch.setattr(services, 'hf_hub_download', fake_hf_hub_download)
	
	downloaded_files = []
	
	def fake_download_file(
		self,
		repo_id: str,
		filename: str,
		revision: str,
		snapshot_dir: str,
		file_index: int,
		file_size: int,
		progress,
		token=None,
	):
		downloaded_files.append(filename)
		local_path = pathlib.Path(snapshot_dir) / filename
		local_path.parent.mkdir(parents=True, exist_ok=True)
		with open(local_path, 'wb') as dest:
			dest.write(b'x' * max(file_size, 1))
			progress.update_bytes(max(file_size, 1))
		return str(local_path)
	
	monkeypatch.setattr(services.DownloadService, 'download_file', fake_download_file, raising=False)
	
	# Act
	service.download_model('test/repo', mock_db)
	
	# Assert: only expected files should be downloaded
	expected_files = {'model_index.json', 'unet/model.safetensors', 'vae/config.json'}
	assert set(downloaded_files) == expected_files


def test_list_files_handles_empty_siblings(monkeypatch: pytest.MonkeyPatch):
	# Arrange
	services = import_services_with_stubs()
	service = services.DownloadService()
	
	class DummyApi:
		def repo_info(self, repo_id: str):
			return SimpleNamespace(siblings=[])
	
	service.api = DummyApi()
	
	# Act
	result = service.list_files('empty/repo')
	
	# Assert
	assert result == []


def test_download_service_init_creates_executor():
	# Arrange & Act
	services = import_services_with_stubs()
	service = services.DownloadService()
	
	# Assert
	assert service.executor is not None
	assert hasattr(service, 'api')


def test_download_model_progress_tracking_increments_correctly(
	tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
	# Arrange
	dummy_socket = DummySocket()
	services = import_services_with_stubs(dummy_socket)
	service = services.DownloadService()
	mock_db = MagicMock()
	
	monkeypatch.setattr(service, 'get_components', lambda _id, revision=None: ['unet'])
	monkeypatch.setattr(service, 'list_files', lambda _id, repo_info=None: [
		'model_index.json',
		'unet/model1.bin',
		'unet/model2.bin',
	])
	monkeypatch.setattr(
		service,
		'get_file_sizes_map',
		lambda _id, repo_info=None: {
			'model_index.json': 5,
			'unet/model1.bin': 10,
			'unet/model2.bin': 15,
		},
	)

	class DummyApi:
		def repo_info(self, repo_id: str):
			return SimpleNamespace(sha='main', siblings=[])

	service.api = DummyApi()  # type: ignore[assignment]
	
	model_root = tmp_path / 'cache-progress'
	monkeypatch.setattr(services.storage_service, 'get_model_dir', lambda _id: str(model_root))
	
	model_index_path = tmp_path / 'model_index.json'
	model_index_path.write_text(json.dumps({'unet': ['cfg']}), encoding='utf-8')
	
	def fake_hf_hub_download(*, repo_id: str, filename: str, **_kwargs):
		return str(model_index_path)
	
	monkeypatch.setattr(services, 'hf_hub_download', fake_hf_hub_download)
	
	def fake_download_file(
		self,
		repo_id: str,
		filename: str,
		revision: str,
		snapshot_dir: str,
		file_index: int,
		file_size: int,
		progress,
		token=None,
	):
		local_path = pathlib.Path(snapshot_dir) / filename
		local_path.parent.mkdir(parents=True, exist_ok=True)
		size = file_size or 4
		with open(local_path, 'wb') as dest:
			for chunk in (max(1, size // 3), max(1, size // 3), size - 2 * max(1, size // 3)):
				if chunk <= 0:
					continue
				dest.write(b'x' * chunk)
				progress.update_bytes(chunk)
		return str(local_path)
	
	monkeypatch.setattr(services.DownloadService, 'download_file', fake_download_file, raising=False)
	
	# Act
	service.download_model('test/repo', mock_db)
	
	# Assert: progress should report cumulative bytes after each file completes
	completions = [call for call in dummy_socket.progress_calls if call[5] == 'file_complete']
	assert [call[1] for call in completions] == [1, 2, 3]
	assert completions[0][3] == 5
	assert completions[1][3] == 15
	assert completions[2][3] == 30
	assert completions[2][4] == 30
