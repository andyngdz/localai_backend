"""Tests for app/database/crud.py delete_model function

Covers:
- Successful model deletion from database and filesystem
- Error handling for non-existent models
- Filesystem cleanup including lock directories
- Database rollback on errors
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session

from app.database.crud import delete_model
from app.database.models.model import Model


class TestDeleteModel:
	"""Test delete_model CRUD function."""

	def setup_method(self):
		"""Setup test method."""
		self.db_mock = MagicMock(spec=Session)
		self.model_id = 'test/model'
		self.model_dir = '/cache/models--test--model'

	def test_delete_model_success(self):
		"""Test successful model deletion."""
		# Setup
		mock_model = Model(model_id=self.model_id, model_dir=self.model_dir)
		self.db_mock.query().filter().first.return_value = mock_model

		with tempfile.TemporaryDirectory() as temp_dir:
			mock_model.model_dir = temp_dir
			
			with patch('app.services.storage_service') as mock_storage_service:
				lock_dir = os.path.join(temp_dir, 'lock')
				os.makedirs(lock_dir)
				mock_storage_service.get_model_lock_dir.return_value = lock_dir

				# Execute
				result = delete_model(self.db_mock, self.model_id)

				# Verify
				assert result == self.model_id
				self.db_mock.delete.assert_called_once_with(mock_model)
				self.db_mock.commit.assert_called_once()
				assert not os.path.exists(lock_dir)

	def test_delete_nonexistent_model(self):
		"""Test deletion fails for non-existent model."""
		# Setup
		self.db_mock.query().filter().first.return_value = None

		# Execute & Verify
		with pytest.raises(ValueError) as exc_info:
			delete_model(self.db_mock, self.model_id)

		assert 'does not exist' in str(exc_info.value)
		self.db_mock.delete.assert_not_called()
		self.db_mock.commit.assert_not_called()

	def test_delete_model_filesystem_error(self):
		"""Test error handling when filesystem deletion fails."""
		# Setup
		mock_model = Model(model_id=self.model_id, model_dir=self.model_dir)
		self.db_mock.query().filter().first.return_value = mock_model

		with patch('shutil.rmtree') as mock_rmtree:
			with patch('os.path.exists') as mock_exists:
				mock_exists.return_value = True  # Pretend directory exists
				mock_rmtree.side_effect = OSError('Permission denied')

				# Execute & Verify
				with pytest.raises(ValueError) as exc_info:
					delete_model(self.db_mock, self.model_id)

				assert 'Error deleting model' in str(exc_info.value)
				self.db_mock.rollback.assert_called_once()

	def test_delete_model_nonexistent_directory(self):
		"""Test deletion when model directory doesn't exist."""
		# Setup
		mock_model = Model(model_id=self.model_id, model_dir='/nonexistent/path')
		self.db_mock.query().filter().first.return_value = mock_model

		with patch('app.services.storage_service') as mock_storage_service:
			mock_storage_service.get_model_lock_dir.return_value = '/nonexistent/lock'

			# Execute
			result = delete_model(self.db_mock, self.model_id)

			# Verify - should still succeed and delete from database
			assert result == self.model_id
			self.db_mock.delete.assert_called_once_with(mock_model)
			self.db_mock.commit.assert_called_once()