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
