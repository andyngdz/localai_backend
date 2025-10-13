from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def mock_socket():
	"""Create mock socket service."""
	with patch('app.features.downloads.progress.socket_service') as mock_service:
		mock_service.download_step_progress = Mock()
		yield mock_service


@pytest.fixture
def mock_logger():
	"""Create mock logger."""
	logger = Mock()
	logger.info = Mock()
	return logger


@pytest.fixture
def progress_instance(mock_socket, mock_logger):
	"""Create DownloadProgress instance with mocked dependencies."""
	from app.features.downloads.progress import DownloadProgress

	progress = DownloadProgress(
		id='test-repo',
		desc='Downloading test-repo',
		file_sizes=[100, 200, 300],
		logger=mock_logger,
		total=3,
		disable=False,  # Enable tqdm counter for testing
	)
	return progress


class TestDownloadProgressInit:
	def test_initializes_with_correct_attributes(self, mock_socket, mock_logger):
		"""Test that DownloadProgress initializes with correct state."""
		from app.features.downloads.progress import DownloadProgress

		progress = DownloadProgress(
			id='test-repo',
			desc='Test download',
			file_sizes=[10, 20, 30],
			logger=mock_logger,
			total=3,
		)

		assert progress.id == 'test-repo'
		assert progress.desc == 'Test download'
		assert progress.file_sizes == [10, 20, 30]
		assert progress.downloaded_size == 0
		assert progress.total_downloaded_size == 60
		assert progress.completed_files_size == 0
		assert progress.current_file is None

		progress.close()

	def test_emits_init_progress_on_creation(self, mock_socket, mock_logger):
		"""Test that init event is emitted during construction."""
		from app.features.downloads.progress import DownloadProgress

		progress = DownloadProgress(
			id='test-repo',
			desc='Test',
			file_sizes=[100],
			logger=mock_logger,
			total=1,
		)

		# Check first call was init
		assert mock_socket.download_step_progress.call_count >= 1
		first_call_args = mock_socket.download_step_progress.call_args_list[0][0][0]
		assert first_call_args.phase == 'init'
		assert first_call_args.id == 'test-repo'

		progress.close()


class TestEmitProgress:
	@pytest.mark.parametrize('phase', ['init', 'file_start', 'file_complete', 'chunk', 'size_update', 'complete'])
	def test_emits_with_correct_phase(self, progress_instance, mock_socket, phase):
		"""Test that emit_progress sends correct phase."""
		mock_socket.download_step_progress.reset_mock()

		progress_instance.emit_progress(phase)

		mock_socket.download_step_progress.assert_called_once()
		call_args = mock_socket.download_step_progress.call_args[0][0]
		assert call_args.phase == phase

	def test_includes_all_required_fields(self, progress_instance, mock_socket):
		"""Test that emitted progress includes all required fields."""
		mock_socket.download_step_progress.reset_mock()
		progress_instance.downloaded_size = 50
		progress_instance.n = 1

		progress_instance.emit_progress('chunk')

		call_args = mock_socket.download_step_progress.call_args[0][0]
		assert call_args.id == 'test-repo'
		assert call_args.step == 1
		assert call_args.total == 3
		assert call_args.downloaded_size == 50
		assert call_args.total_downloaded_size == 600
		assert call_args.phase == 'chunk'

	def test_uses_provided_current_file_over_instance_attribute(self, progress_instance, mock_socket):
		"""Test that explicit current_file parameter takes precedence."""
		mock_socket.download_step_progress.reset_mock()
		progress_instance.current_file = 'default.bin'

		progress_instance.emit_progress('file_start', current_file='override.bin')

		call_args = mock_socket.download_step_progress.call_args[0][0]
		assert call_args.current_file == 'override.bin'

	@pytest.mark.parametrize('phase', ['file_start', 'file_complete'])
	def test_logs_for_file_events(self, progress_instance, mock_logger, phase):
		"""Test that file_start and file_complete events trigger logging."""
		mock_logger.info.reset_mock()
		progress_instance.n = 2

		progress_instance.emit_progress(phase)

		# tqdm might override desc, so just verify logging was called
		mock_logger.info.assert_called_once()
		args = mock_logger.info.call_args[0]
		assert args[0] == '%s %s/%s'
		assert args[2] == 2
		assert args[3] == 3

	def test_does_not_log_for_non_file_events(self, progress_instance, mock_logger):
		"""Test that chunk/size_update events don't trigger logging."""
		mock_logger.info.reset_mock()

		progress_instance.emit_progress('chunk')

		mock_logger.info.assert_not_called()


class TestStartFile:
	def test_sets_current_file(self, progress_instance):
		"""Test that start_file updates current_file attribute."""
		progress_instance.start_file('model.bin')

		assert progress_instance.current_file == 'model.bin'

	def test_emits_file_start_event(self, progress_instance, mock_socket):
		"""Test that start_file emits file_start progress."""
		mock_socket.download_step_progress.reset_mock()

		progress_instance.start_file('model.bin')

		call_args = mock_socket.download_step_progress.call_args[0][0]
		assert call_args.phase == 'file_start'
		assert call_args.current_file == 'model.bin'


class TestSetFileSize:
	def test_updates_file_size_and_total(self, progress_instance):
		"""Test that set_file_size updates both file_sizes and total_downloaded_size."""
		initial_total = progress_instance.total_downloaded_size

		progress_instance.set_file_size(0, 150)

		assert progress_instance.file_sizes[0] == 150
		assert progress_instance.total_downloaded_size == initial_total + 50

	def test_updates_completed_files_size_for_completed_files(self, progress_instance):
		"""Test that set_file_size updates completed_files_size for already completed files."""
		progress_instance.n = 2
		progress_instance.completed_files_size = 300

		progress_instance.set_file_size(0, 150)

		assert progress_instance.completed_files_size == 350

	def test_ignores_negative_index(self, progress_instance):
		"""Test that negative index is ignored."""
		initial_sizes = progress_instance.file_sizes.copy()

		progress_instance.set_file_size(-1, 999)

		assert progress_instance.file_sizes == initial_sizes

	def test_ignores_out_of_bounds_index(self, progress_instance):
		"""Test that out of bounds index is ignored."""
		initial_sizes = progress_instance.file_sizes.copy()

		progress_instance.set_file_size(10, 999)

		assert progress_instance.file_sizes == initial_sizes

	def test_ignores_when_size_unchanged(self, progress_instance, mock_socket):
		"""Test that no update occurs when size is the same."""
		mock_socket.download_step_progress.reset_mock()

		progress_instance.set_file_size(0, 100)

		mock_socket.download_step_progress.assert_not_called()

	def test_clamps_negative_total_to_zero(self, progress_instance):
		"""Test that negative total_downloaded_size is clamped to 0."""
		progress_instance.total_downloaded_size = 50

		progress_instance.set_file_size(0, 0)

		assert progress_instance.total_downloaded_size == 0

	def test_clamps_negative_size_input_to_zero(self, progress_instance):
		"""Test that negative size input is clamped to 0."""
		progress_instance.set_file_size(0, -100)

		assert progress_instance.file_sizes[0] == 0

	def test_emits_size_update_event(self, progress_instance, mock_socket):
		"""Test that set_file_size emits size_update progress."""
		mock_socket.download_step_progress.reset_mock()

		progress_instance.set_file_size(0, 150)

		call_args = mock_socket.download_step_progress.call_args[0][0]
		assert call_args.phase == 'size_update'


class TestUpdateBytes:
	def test_increments_downloaded_size(self, progress_instance):
		"""Test that update_bytes increments downloaded_size."""
		initial = progress_instance.downloaded_size

		progress_instance.update_bytes(50)

		assert progress_instance.downloaded_size == initial + 50

	def test_ignores_zero_or_negative_bytes(self, progress_instance):
		"""Test that zero or negative byte counts are ignored."""
		initial = progress_instance.downloaded_size

		progress_instance.update_bytes(0)
		progress_instance.update_bytes(-10)

		assert progress_instance.downloaded_size == initial

	def test_clamps_to_total_downloaded_size(self, progress_instance):
		"""Test that downloaded_size doesn't exceed total_downloaded_size."""
		progress_instance.total_downloaded_size = 100

		progress_instance.update_bytes(150)

		assert progress_instance.downloaded_size == 100

	def test_emits_chunk_event(self, progress_instance, mock_socket):
		"""Test that update_bytes emits chunk progress."""
		mock_socket.download_step_progress.reset_mock()

		progress_instance.update_bytes(25)

		call_args = mock_socket.download_step_progress.call_args[0][0]
		assert call_args.phase == 'chunk'
		assert call_args.downloaded_size == 25


class TestUpdate:
	def test_increments_completed_files_size(self, progress_instance):
		"""Test that update increments completed_files_size by sum of completed files."""
		progress_instance.update(1)

		assert progress_instance.completed_files_size == 100

		progress_instance.update(1)

		assert progress_instance.completed_files_size == 300

	def test_adjusts_downloaded_size_to_match_completed_files(self, progress_instance):
		"""Test that downloaded_size is updated to match completed_files_size."""
		progress_instance.downloaded_size = 50

		progress_instance.update(1)

		assert progress_instance.downloaded_size == 100

	def test_does_not_decrease_downloaded_size(self, progress_instance):
		"""Test that downloaded_size doesn't decrease if already ahead."""
		progress_instance.downloaded_size = 500

		progress_instance.update(1)

		assert progress_instance.downloaded_size == 500

	def test_handles_multiple_file_update(self, progress_instance):
		"""Test that updating by n>1 processes multiple files."""
		progress_instance.update(2)

		assert progress_instance.completed_files_size == 300
		assert progress_instance.n == 2

	def test_handles_update_beyond_file_count(self, progress_instance):
		"""Test that update handles n exceeding file_sizes length."""
		progress_instance.update(10)

		assert progress_instance.completed_files_size == 600

	def test_emits_file_complete_event(self, progress_instance, mock_socket):
		"""Test that update emits file_complete progress."""
		mock_socket.download_step_progress.reset_mock()

		progress_instance.update(1)

		call_args = mock_socket.download_step_progress.call_args[0][0]
		assert call_args.phase == 'file_complete'


class TestClose:
	def test_emits_complete_event(self, progress_instance, mock_socket):
		"""Test that close emits complete progress."""
		mock_socket.download_step_progress.reset_mock()

		progress_instance.close()

		call_args = mock_socket.download_step_progress.call_args[0][0]
		assert call_args.phase == 'complete'

	def test_calls_parent_close(self, progress_instance):
		"""Test that close calls tqdm's close method."""
		with patch('tqdm.tqdm.close') as mock_parent_close:
			progress_instance.close()
			mock_parent_close.assert_called_once()


class TestIntegrationScenarios:
	def test_complete_download_flow(self, mock_socket, mock_logger):
		"""Test a realistic download scenario with multiple files."""
		from app.features.downloads.progress import DownloadProgress

		progress = DownloadProgress(
			id='test/model',
			desc='Downloading',
			file_sizes=[100, 200],
			logger=mock_logger,
			total=2,
			disable=False,
		)

		# Start first file
		progress.start_file('file1.bin')
		progress.update_bytes(50)
		progress.update_bytes(50)
		progress.update(1)

		assert progress.n == 1
		assert progress.downloaded_size == 100
		assert progress.completed_files_size == 100

		# Start second file
		progress.start_file('file2.bin')
		progress.update_bytes(100)
		progress.update_bytes(100)
		progress.update(1)

		assert progress.n == 2
		assert progress.downloaded_size == 300
		assert progress.completed_files_size == 300

		# Close
		progress.close()

		# Verify all events were emitted
		calls = [call[0][0].phase for call in mock_socket.download_step_progress.call_args_list]
		assert 'init' in calls
		assert calls.count('file_start') == 2
		assert calls.count('chunk') == 4
		assert calls.count('file_complete') == 2
		assert 'complete' in calls

	def test_file_size_adjustment_mid_download(self, mock_socket, mock_logger):
		"""Test adjusting file size during download (e.g., from Content-Length header)."""
		from app.features.downloads.progress import DownloadProgress

		progress = DownloadProgress(
			id='test/model',
			desc='Downloading',
			file_sizes=[0, 0],
			logger=mock_logger,
			total=2,
			disable=False,
		)

		# Initial total is 0
		assert progress.total_downloaded_size == 0

		# Set actual size from server
		progress.set_file_size(0, 100)
		assert progress.total_downloaded_size == 100

		progress.start_file('file1.bin')
		progress.update_bytes(50)
		progress.update_bytes(50)
		progress.update(1)

		# Set second file size
		progress.set_file_size(1, 200)
		assert progress.total_downloaded_size == 300

		progress.start_file('file2.bin')
		progress.update_bytes(200)
		progress.update(1)

		assert progress.downloaded_size == 300
		assert progress.completed_files_size == 300
