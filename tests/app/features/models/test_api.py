"""Tests for app/features/models/api.py

Covers:
- Delete model endpoint functionality
- Model in use prevention
- Database and filesystem cleanup
- Error handling for non-existent models
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.features.models.api import delete_model
from app.schemas.responses import JSONResponseMessage


class TestDeleteModelEndpoint:
	"""Test delete model endpoint."""

	def setup_method(self):
		"""Setup test method."""
		self.db_mock = MagicMock(spec=Session)
		self.model_id = 'test/model'

	@patch('app.features.models.api.model_manager')
	@patch('app.features.models.api.delete_model')
	def test_delete_model_success(self, mock_delete_model, mock_model_manager):
		"""Test successful model deletion."""
		# Setup
		mock_model_manager.id = 'different/model'  # Model not in use
		mock_delete_model.return_value = self.model_id

		# Execute
		result = delete_model(self.model_id, self.db_mock)

		# Verify
		mock_delete_model.assert_called_once_with(self.db_mock, self.model_id)
		assert isinstance(result, JSONResponseMessage)
		# Check the content dict that was passed to JSONResponse
		assert result.body.decode() == '{"message":"Model test/model deleted successfully"}'

	@patch('app.features.models.api.model_manager')
	def test_delete_model_in_use(self, mock_model_manager):
		"""Test deletion fails when model is currently loaded."""
		# Setup
		mock_model_manager.id = self.model_id  # Model is in use

		# Execute & Verify
		with pytest.raises(HTTPException) as exc_info:
			delete_model(self.model_id, self.db_mock)

		assert exc_info.value.status_code == status.HTTP_409_CONFLICT
		assert 'Model is currently loaded' in str(exc_info.value.detail)

	@patch('app.features.models.api.model_manager')
	@patch('app.features.models.api.delete_model')
	def test_delete_nonexistent_model(self, mock_delete_model, mock_model_manager):
		"""Test deletion fails for non-existent model."""
		# Setup
		mock_model_manager.id = 'different/model'
		mock_delete_model.side_effect = ValueError(f'Model with id {self.model_id} does not exist.')

		# Execute & Verify
		with pytest.raises(HTTPException) as exc_info:
			delete_model(self.model_id, self.db_mock)

		assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
		assert 'does not exist' in str(exc_info.value.detail)

	@patch('app.features.models.api.model_manager')
	@patch('app.features.models.api.delete_model')
	def test_delete_model_general_error(self, mock_delete_model, mock_model_manager):
		"""Test general error handling during deletion."""
		# Setup
		mock_model_manager.id = 'different/model'
		mock_delete_model.side_effect = Exception('Filesystem error')

		# Execute & Verify
		with pytest.raises(HTTPException) as exc_info:
			delete_model(self.model_id, self.db_mock)

		assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
		assert 'Failed to delete model' in str(exc_info.value.detail)
