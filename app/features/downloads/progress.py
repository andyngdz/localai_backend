from typing import Optional

from tqdm import tqdm as BaseTqdm

from app.socket import socket_service

from .schemas import DownloadStepProgressResponse


class DownloadProgress(BaseTqdm):
	"""
	tqdm wrapper that emits websocket payloads describing byte-level progress.

	A download remains at the same step until a file completes. Each chunk emit
	carries cumulative bytes so the UI can render an accurate percentage.
	"""

	def __init__(self, *args, **kwargs):
		self.id = kwargs.pop('id')
		self.desc = kwargs.pop('desc')
		self.file_sizes = kwargs.pop('file_sizes')
		self.logger = kwargs.pop('logger')
		self.downloaded_size = 0
		self.total_downloaded_size = sum(self.file_sizes)
		self.completed_files_size = 0
		self.current_file: Optional[str] = None

		kwargs.setdefault('disable', None)
		kwargs.setdefault('mininterval', 0.25)
		kwargs.setdefault('leave', False)

		super().__init__(*args, **kwargs)
		self.emit_progress('init')

	def emit_progress(self, phase: str, *, current_file: Optional[str] = None) -> None:
		"""Push the latest cumulative byte totals to all websocket clients."""
		socket_service.download_step_progress(
			DownloadStepProgressResponse(
				id=self.id,
				step=self.n,
				total=self.total,
				downloaded_size=self.downloaded_size,
				total_downloaded_size=self.total_downloaded_size,
				phase=phase,
				current_file=current_file or self.current_file,
			)
		)

		if phase in {'file_start', 'file_complete'}:
			self.logger.info('%s %s/%s', self.desc, self.n, self.total)

	def start_file(self, filename: str) -> None:
		"""Mark the beginning of a file download so the UI can show file-level progress."""
		self.current_file = filename
		self.emit_progress('file_start', current_file=filename)

	def set_file_size(self, index: int, size: int) -> None:
		"""Update the recorded size for a file and adjust totals accordingly."""
		if index < 0 or index >= len(self.file_sizes):
			return

		size = max(size, 0)
		previous = self.file_sizes[index]

		if previous == size:
			return

		delta = size - previous
		self.file_sizes[index] = size
		self.total_downloaded_size += delta
		if self.total_downloaded_size < 0:
			self.total_downloaded_size = 0

		if index < self.n:
			self.completed_files_size += delta
			if self.completed_files_size < 0:
				self.completed_files_size = 0

		self.emit_progress('size_update')

	def update_bytes(self, byte_count: int) -> None:
		"""Increment downloaded bytes and emit progress for partial file download."""
		if byte_count <= 0:
			return

		self.downloaded_size += byte_count
		if self.total_downloaded_size > 0 and self.downloaded_size > self.total_downloaded_size:
			self.downloaded_size = self.total_downloaded_size
		self.emit_progress('chunk')

	def update(self, n=1):
		super().update(n)
		for i in range(max(0, self.n - n), self.n):
			if i < len(self.file_sizes):
				self.completed_files_size += self.file_sizes[i]

		if self.downloaded_size < self.completed_files_size:
			self.downloaded_size = self.completed_files_size

		self.emit_progress('file_complete')

	def close(self):
		"""Close the progress bar and emit final completion event."""
		self.emit_progress('complete')
		super().close()
