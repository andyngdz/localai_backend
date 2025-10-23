import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests
from requests import Session


@pytest.fixture
def mock_service():
	"""Create DownloadService with mocked dependencies."""
	with (
		patch('app.features.downloads.services.HfApi') as mock_api,
		patch('app.features.downloads.services.model_service') as mock_model_service,
		patch('app.features.downloads.services.storage_service') as mock_storage_service,
	):
		mock_storage_service.get_model_dir.return_value = '/tmp/test-models'

		from app.features.downloads.services import DownloadService

		service = DownloadService()
		service.api = mock_api
		service.session = MagicMock(spec=Session)

		yield service, mock_model_service, mock_storage_service


@pytest.fixture
def mock_progress():
	"""Create mock progress tracker."""
	progress = Mock()
	progress.set_file_size = Mock()
	progress.update_bytes = Mock()
	progress.start_file = Mock()
	progress.update = Mock()
	progress.close = Mock()
	return progress


class TestDownloadServiceInit:
	def test_creates_executor_and_api(self, mock_service):
		service, _, _ = mock_service
		assert service.executor is not None
		assert hasattr(service, 'api')


class TestDownloadServiceStart:
	@pytest.mark.asyncio
	async def test_invokes_download_model_in_executor(self, mock_service):
		service, _, _ = mock_service
		service.download_model = Mock(return_value='/fake/path')
		mock_db = Mock()

		result = await service.start('test/repo', mock_db)

		assert result == '/fake/path'
		service.download_model.assert_called_once_with('test/repo', mock_db)


class TestDownloadModelValidation:
	@pytest.mark.parametrize(
		'model_id,error_msg',
		[
			('', 'Model ID cannot be empty'),
			('   ', 'Model ID cannot be empty'),
		],
	)
	def test_raises_error_for_invalid_model_id(self, mock_service, model_id, error_msg):
		service, _, _ = mock_service
		with pytest.raises(ValueError, match=error_msg):
			service.download_model(model_id, Mock())


class TestDownloadModel:
	def test_returns_none_when_no_files_to_download(self, mock_service, monkeypatch):
		service, _, _ = mock_service
		service.api.repo_info.return_value = SimpleNamespace(sha='main', siblings=[])
		monkeypatch.setattr(service, 'get_components', lambda *args, **kwargs: [])
		monkeypatch.setattr(service, 'list_files', lambda *args, **kwargs: [])

		result = service.download_model('test/repo', Mock())

		assert result is None

	def test_logs_error_when_database_save_fails(self, mock_service, tmp_path, monkeypatch):
		service, mock_model_service, mock_storage_service = mock_service
		mock_storage_service.get_model_dir.return_value = str(tmp_path)
		mock_model_service.add_model.side_effect = ValueError('DB error')

		model_index = tmp_path / 'model_index.json'
		model_index.write_text(json.dumps({'unet': ['cfg']}))

		service.api.repo_info.return_value = SimpleNamespace(
			sha='main', siblings=[SimpleNamespace(rfilename='unet/model.bin', size=10)]
		)
		monkeypatch.setattr(service, 'get_components', lambda *args, **kwargs: ['unet'])
		monkeypatch.setattr(service, 'list_files', lambda *args, **kwargs: ['unet/model.bin'])

		def fake_download(**kwargs):
			path = tmp_path / 'snapshots' / 'main' / kwargs['filename']
			path.parent.mkdir(parents=True, exist_ok=True)
			path.write_bytes(b'test')
			return str(path)

		monkeypatch.setattr(service, 'download_file', fake_download)

		with (
			patch('app.features.downloads.services.logger') as mock_logger,
			patch('app.features.downloads.services.hf_hub_download', return_value=str(model_index)),
		):
			result = service.download_model('test/repo', Mock())
			assert result is not None
			mock_logger.error.assert_called_once()

	def test_closes_progress_on_exception(self, mock_service, tmp_path, monkeypatch):
		service, _, mock_storage_service = mock_service
		mock_storage_service.get_model_dir.return_value = str(tmp_path)

		model_index = tmp_path / 'model_index.json'
		model_index.write_text(json.dumps({'unet': ['cfg']}))

		service.api.repo_info.return_value = SimpleNamespace(
			sha='main', siblings=[SimpleNamespace(rfilename='unet/model.bin', size=10)]
		)
		monkeypatch.setattr(service, 'get_components', lambda *args, **kwargs: ['unet'])
		monkeypatch.setattr(service, 'list_files', lambda *args, **kwargs: ['unet/model.bin'])
		monkeypatch.setattr(service, 'download_file', Mock(side_effect=ConnectionError('Failed')))

		mock_progress = Mock()
		with (
			patch('app.features.downloads.services.DownloadTqdm', return_value=mock_progress),
			patch('app.features.downloads.services.hf_hub_download', return_value=str(model_index)),
		):
			with pytest.raises(ConnectionError):
				service.download_model('test/repo', Mock())

		mock_progress.close.assert_called_once()


class TestGetIgnoreComponents:
	@pytest.mark.parametrize(
		'files,scopes,expected',
		[
			(['unet/model.bin', 'unet/model.safetensors', 'vae/model.bin'], ['unet/*', 'vae/*'], ['unet/model.bin']),
			(['unet/model.bin', 'vae/model.bin'], ['unet/*', 'vae/*'], []),
			([], ['unet/*'], []),
			(['out_of_scope/model.bin', 'out_of_scope/model.safetensors'], ['unet/*'], []),
		],
	)
	def test_filters_bin_files_correctly(self, mock_service, files, scopes, expected):
		service, _, _ = mock_service
		result = service.get_ignore_components(files, scopes)
		assert result == expected


class TestListFiles:
	def test_returns_filenames_from_siblings(self, mock_service):
		service, _, _ = mock_service
		service.api.repo_info.return_value = SimpleNamespace(
			siblings=[
				SimpleNamespace(rfilename='a.txt'),
				SimpleNamespace(rfilename='b/c.bin'),
			]
		)

		result = service.list_files('test/repo')

		assert result == ['a.txt', 'b/c.bin']

	def test_handles_empty_siblings(self, mock_service):
		service, _, _ = mock_service
		service.api.repo_info.return_value = SimpleNamespace(siblings=[])

		result = service.list_files('test/repo')

		assert result == []


class TestGetFileSizesMap:
	def test_returns_size_dict(self, mock_service):
		service, _, _ = mock_service
		service.api.repo_info.return_value = SimpleNamespace(
			siblings=[
				SimpleNamespace(rfilename='file1.bin', size=100),
				SimpleNamespace(rfilename='file2.bin', size=200),
			]
		)

		result = service.get_file_sizes_map('test/repo')

		assert result == {'file1.bin': 100, 'file2.bin': 200}

	def test_handles_missing_size_attribute(self, mock_service):
		service, _, _ = mock_service
		service.api.repo_info.return_value = SimpleNamespace(
			siblings=[
				SimpleNamespace(rfilename='file1.bin', size=100),
				SimpleNamespace(rfilename='file2.bin'),
				SimpleNamespace(rfilename='file3.bin', size=None),
			]
		)

		result = service.get_file_sizes_map('test/repo')

		assert result == {'file1.bin': 100, 'file2.bin': 0, 'file3.bin': 0}


class TestGetComponents:
	def test_parses_model_index_json(self, mock_service, tmp_path):
		service, _, _ = mock_service
		model_index = tmp_path / 'model_index.json'
		model_index.write_text(
			json.dumps(
				{
					'unet': ['config'],
					'vae': [None],
					'scheduler': 'not-a-list',
				}
			)
		)

		with patch('app.features.downloads.services.hf_hub_download', return_value=str(model_index)):
			result = service.get_components('test/repo')

		assert result == ['unet']

	def test_handles_malformed_json(self, mock_service, tmp_path):
		service, _, _ = mock_service
		model_index = tmp_path / 'model_index.json'
		model_index.write_text('invalid json')

		with patch('app.features.downloads.services.hf_hub_download', return_value=str(model_index)):
			with pytest.raises(json.JSONDecodeError):
				service.get_components('test/repo')


class TestAuthHeaders:
	@pytest.mark.parametrize(
		'token,expected',
		[
			(None, {}),
			('test-token-123', {'Authorization': 'Bearer test-token-123'}),
		],
	)
	def test_builds_headers_correctly(self, mock_service, token, expected):
		service, _, _ = mock_service
		result = service.auth_headers(token)
		assert result == expected


class TestDownloadFileSecurity:
	@pytest.mark.parametrize(
		'filename',
		[
			'../../../etc/passwd',
			'/etc/passwd',
		],
	)
	def test_blocks_path_traversal(self, mock_service, mock_progress, filename):
		service, _, _ = mock_service

		with pytest.raises(ValueError, match='attempts to escape snapshot directory'):
			service.download_file(
				repo_id='test/repo',
				filename=filename,
				revision='main',
				snapshot_dir='/tmp/test',
				file_index=0,
				progress=mock_progress,
			)


class TestDownloadFile:
	def test_skips_when_file_exists(self, mock_service, mock_progress, tmp_path):
		service, _, _ = mock_service
		snapshot_dir = tmp_path / 'snapshots'
		snapshot_dir.mkdir()
		existing = snapshot_dir / 'model.bin'
		existing.write_bytes(b'existing')

		with patch('app.features.downloads.services.logger'):
			result = service.download_file(
				repo_id='test/repo',
				filename='model.bin',
				revision='main',
				snapshot_dir=str(snapshot_dir),
				file_index=0,
				progress=mock_progress,
				file_size=5,
			)

		assert result == str(existing)
		mock_progress.set_file_size.assert_called_once_with(0, 8)
		mock_progress.update_bytes.assert_not_called()

	def test_removes_zero_size_files(self, mock_service, mock_progress, tmp_path):
		service, _, _ = mock_service
		snapshot_dir = tmp_path / 'snapshots'
		snapshot_dir.mkdir()
		zero_file = snapshot_dir / 'model.bin'
		zero_file.write_bytes(b'')

		mock_response = Mock()
		mock_response.headers.get.return_value = '5'
		mock_response.iter_content.return_value = [b'hello']
		mock_response.__enter__ = lambda self: mock_response
		mock_response.__exit__ = lambda *args: None

		service.session.get.return_value = mock_response
		result = service.download_file(
			repo_id='test/repo',
			filename='model.bin',
			revision='main',
			snapshot_dir=str(snapshot_dir),
			file_index=0,
			progress=mock_progress,
			file_size=8,
		)

		assert Path(result).read_bytes() == b'hello'

	def test_downloads_successfully(self, mock_service, mock_progress, tmp_path):
		service, _, _ = mock_service
		snapshot_dir = tmp_path / 'snapshots'
		snapshot_dir.mkdir()

		mock_response = Mock()
		mock_response.headers.get.return_value = '10'
		mock_response.iter_content.return_value = [b'hello', b'world']
		mock_response.__enter__ = lambda self: mock_response
		mock_response.__exit__ = lambda *args: None

		service.session.get.return_value = mock_response
		result = service.download_file(
			repo_id='test/repo',
			filename='model.bin',
			revision='main',
			snapshot_dir=str(snapshot_dir),
			file_index=0,
			progress=mock_progress,
			file_size=8,
		)

		assert Path(result).read_bytes() == b'helloworld'
		assert mock_progress.update_bytes.call_count == 2

	def test_cleans_up_part_file_on_error(self, mock_service, mock_progress, tmp_path):
		service, _, _ = mock_service
		snapshot_dir = tmp_path / 'snapshots'
		snapshot_dir.mkdir()

		mock_response = Mock()
		mock_response.headers.get.return_value = '10'
		mock_response.iter_content.side_effect = ConnectionError('Network error')
		mock_response.__enter__ = lambda self: mock_response
		mock_response.__exit__ = lambda *args: None

		service.session.get.return_value = mock_response
		with pytest.raises(ConnectionError):
			service.download_file(
				repo_id='test/repo',
				filename='model.bin',
				revision='main',
				snapshot_dir=str(snapshot_dir),
				file_index=0,
				progress=mock_progress,
				file_size=10,
			)

		assert not (snapshot_dir / 'model.bin.part').exists()


class TestFetchRemoteFileSize:
	def test_returns_content_length(self, mock_service):
		service, _, _ = mock_service
		mock_response = Mock()
		mock_response.headers.get.return_value = '12345'

		service.session.head.return_value = mock_response
		size = service.fetch_remote_file_size('test/repo', 'model.bin', 'main')

		assert size == 12345

	@pytest.mark.parametrize(
		'header_value,expected',
		[
			(None, 0),
			('invalid', 0),
			('-100', 0),
		],
	)
	def test_handles_invalid_content_length(self, mock_service, header_value, expected):
		service, _, _ = mock_service
		mock_response = Mock()
		mock_response.headers.get.return_value = header_value

		service.session.head.return_value = mock_response
		size = service.fetch_remote_file_size('test/repo', 'model.bin', 'main')

		assert size == expected

	def test_handles_http_error(self, mock_service):
		service, _, _ = mock_service

		service.session.head.side_effect = requests.RequestException('HTTP 404')
		size = service.fetch_remote_file_size('test/repo', 'model.bin', 'main')

		assert size == 0
