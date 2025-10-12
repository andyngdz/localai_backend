import sys
from types import ModuleType
from typing import List

import pytest


@pytest.fixture(autouse=True)
def stub_progress_deps():
	if 'app.services.styles' not in sys.modules:
		styles_mod = ModuleType('app.services.styles')

		class StylesService:
			def __init__(self) -> None:
				...

		setattr(styles_mod, 'styles_service', StylesService())
		sys.modules['app.services.styles'] = styles_mod

	if 'app.services.models' not in sys.modules:
		models_mod = ModuleType('app.services.models')

		class ModelService:
			def add_model(self, *_args, **_kwargs):
				...

		setattr(models_mod, 'model_service', ModelService())
		sys.modules['app.services.models'] = models_mod

	if 'app.features.downloads.api' not in sys.modules:
		api_mod = ModuleType('app.features.downloads.api')
		setattr(api_mod, 'downloads', object())
		sys.modules['app.features.downloads.api'] = api_mod


@pytest.fixture
def patched_socket(monkeypatch: pytest.MonkeyPatch):
	from app.features.downloads import progress as progress_module

	socket = DummySocket()
	monkeypatch.setattr(progress_module, 'socket_service', socket)
	return socket


class DummySocket:
	def __init__(self) -> None:
		self.events = []

	def download_step_progress(self, data) -> None:
		self.events.append(data)


class DummyLogger:
	def __init__(self) -> None:
		self.messages: List[str] = []

	def info(self, message: str, *args) -> None:
		self.messages.append(message % args if args else message)


def test_download_progress_captures_phases(patched_socket):
	from app.features.downloads.progress import DownloadProgress

	logger = DummyLogger()
	progress = DownloadProgress(
		id='model',
		desc='Downloading model',
		file_sizes=[10, 20],
		total=2,
		unit='files',
		logger=logger,
	)

	progress.start_file('a.bin')
	progress.update_bytes(4)
	progress.update_bytes(6)
	progress.update(1)

	progress.start_file('b.bin')
	progress.update_bytes(20)
	progress.update(1)

	phases = [event.phase for event in patched_socket.events]
	assert phases.count('file_start') == 2
	assert phases.count('chunk') >= 1
	assert phases.count('file_complete') == 2
	assert patched_socket.events[-1].downloaded_size == 30


def test_download_progress_emits_complete_event_on_close(patched_socket):
	"""Test that close() emits a 'complete' phase event."""
	from app.features.downloads.progress import DownloadProgress

	logger = DummyLogger()
	progress = DownloadProgress(
		id='model',
		desc='Downloading model',
		file_sizes=[10],
		total=1,
		unit='files',
		logger=logger,
	)

	progress.start_file('test.bin')
	progress.update(1)
	progress.close()

	# Check that 'complete' event was emitted
	phases = [event.phase for event in patched_socket.events]
	assert 'complete' in phases
	assert phases[-1] == 'complete'


def test_download_progress_uses_running_total_for_performance(patched_socket):
	"""Test that progress tracking uses O(1) completed_files_size instead of O(n) sum()."""
	from app.features.downloads.progress import DownloadProgress

	logger = DummyLogger()
	progress = DownloadProgress(
		id='model',
		desc='Downloading model',
		file_sizes=[10, 20, 30, 40],
		total=4,
		unit='files',
		logger=logger,
	)

	# Verify completed_files_size is initialized
	assert hasattr(progress, 'completed_files_size')
	assert progress.completed_files_size == 0

	# Complete first file
	progress.start_file('file1.bin')
	progress.update(1)
	assert progress.completed_files_size == 10
	assert progress.downloaded_size >= 10

	# Complete second file
	progress.start_file('file2.bin')
	progress.update(1)
	assert progress.completed_files_size == 30  # 10 + 20
	assert progress.downloaded_size >= 30

	# Complete third file
	progress.start_file('file3.bin')
	progress.update(1)
	assert progress.completed_files_size == 60  # 10 + 20 + 30
	assert progress.downloaded_size >= 60


def test_download_progress_set_file_size_updates_completed_total(patched_socket):
	"""Test that set_file_size() updates completed_files_size for already completed files."""
	from app.features.downloads.progress import DownloadProgress

	logger = DummyLogger()
	progress = DownloadProgress(
		id='model',
		desc='Downloading model',
		file_sizes=[10, 20, 30],
		total=3,
		unit='files',
		logger=logger,
	)

	# Complete first file
	progress.start_file('file1.bin')
	progress.update(1)
	assert progress.completed_files_size == 10

	# Update the size of the first file (already completed)
	progress.set_file_size(0, 15)  # Change from 10 to 15
	assert progress.completed_files_size == 15  # Should be updated
	assert progress.total_downloaded_size == 65  # 15 + 20 + 30

	# Complete second file
	progress.start_file('file2.bin')
	progress.update(1)
	assert progress.completed_files_size == 35  # 15 + 20
