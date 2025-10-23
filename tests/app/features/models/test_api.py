"""Tests for app/features/models/api.py

Covers:
- Delete model endpoint functionality
- Model in use prevention
- Database and filesystem cleanup
- Error handling for non-existent models
- Load model endpoint functionality
- Unload model endpoint functionality
- Model recommendations endpoint functionality
- Model availability checking endpoint functionality
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Response, status
from sqlalchemy.orm import Session

from app.cores.model_loader.cancellation import CancellationException
from app.features.models.api import (
	delete_model_by_id,
	get_model_recommendations,
	is_model_available,
	load_model,
	unload_model,
)
from app.features.models.schemas import (
	LoadModelRequest,
	LoadModelResponse,
	ModelAvailableResponse,
)
from app.schemas.responses import JSONResponseMessage


class TestDeleteModelEndpoint:
	"""Test delete model endpoint."""

	def setup_method(self):
		"""Setup test method."""
		self.db_mock = MagicMock(spec=Session)
		self.model_id = 'test/model'

	@patch('app.features.models.api.model_manager')
	@patch('app.features.models.api.model_service')
	def test_delete_model_success(self, mock_model_service, mock_model_manager):
		"""Test successful model deletion."""
		# Setup
		mock_model_manager.id = 'different/model'  # Model not in use
		mock_model_service.delete_model.return_value = self.model_id

		# Execute
		result = delete_model_by_id(self.model_id, self.db_mock)

		# Verify
		mock_model_service.delete_model.assert_called_once_with(self.db_mock, self.model_id)
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
			delete_model_by_id(self.model_id, self.db_mock)

		assert exc_info.value.status_code == status.HTTP_409_CONFLICT
		assert 'Model is currently loaded' in str(exc_info.value.detail)

	@patch('app.features.models.api.model_manager')
	@patch('app.features.models.api.model_service')
	def test_delete_nonexistent_model(self, mock_model_service, mock_model_manager):
		"""Test deletion fails for non-existent model."""
		# Setup
		mock_model_manager.id = 'different/model'
		mock_model_service.delete_model.side_effect = ValueError(f'Model with id {self.model_id} does not exist.')

		# Execute & Verify
		with pytest.raises(HTTPException) as exc_info:
			delete_model_by_id(self.model_id, self.db_mock)

		assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
		assert 'does not exist' in str(exc_info.value.detail)

	@patch('app.features.models.api.model_manager')
	@patch('app.features.models.api.model_service')
	def test_delete_model_general_error(self, mock_model_service, mock_model_manager):
		"""Test general error handling during deletion."""
		# Setup
		mock_model_manager.id = 'different/model'
		mock_model_service.delete_model.side_effect = Exception('Filesystem error')

		# Execute & Verify
		with pytest.raises(HTTPException) as exc_info:
			delete_model_by_id(self.model_id, self.db_mock)

		assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
		assert 'Failed to delete model' in str(exc_info.value.detail)


class TestLoadModelEndpoint:
	"""Test load model endpoint."""

	def setup_method(self):
		"""Setup test method."""
		self.model_id = 'test/model'
		self.request = LoadModelRequest(id=self.model_id)
		self.model_config = {'model_type': 'diffusion', 'version': '1.0'}
		self.sample_size = 512

	@pytest.mark.asyncio
	@patch('app.features.models.api.model_manager')
	async def test_load_model_success(self, mock_model_manager):
		"""Test successful model loading."""
		# Arrange
		# Create a proper AsyncMock for the async method
		mock_model_manager.load_model_async = AsyncMock(return_value=self.model_config)
		mock_model_manager.get_sample_size.return_value = self.sample_size

		# Act
		result = await load_model(self.request)

		# Assert
		mock_model_manager.load_model_async.assert_called_once_with(self.model_id)
		mock_model_manager.get_sample_size.assert_called_once()
		assert isinstance(result, LoadModelResponse)
		assert result.id == self.model_id
		assert result.config == self.model_config
		assert result.sample_size == self.sample_size

	@pytest.mark.asyncio
	@patch('app.features.models.api.model_manager')
	async def test_load_model_file_not_found(self, mock_model_manager):
		"""Test model loading fails when file not found."""
		# Arrange
		mock_model_manager.load_model_async = AsyncMock(side_effect=FileNotFoundError('Model file not found'))

		# Act & Assert
		with pytest.raises(HTTPException) as exc_info:
			await load_model(self.request)

		assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
		assert 'Model files not found' in str(exc_info.value.detail)

	@pytest.mark.asyncio
	@patch('app.features.models.api.model_manager')
	async def test_load_model_general_error(self, mock_model_manager):
		"""Test general error handling during model loading."""
		# Arrange
		mock_model_manager.load_model_async = AsyncMock(side_effect=Exception('Failed to load model'))

		# Act & Assert
		with pytest.raises(HTTPException) as exc_info:
			await load_model(self.request)

		assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
		assert 'Failed to load model' in str(exc_info.value.detail)

	@pytest.mark.asyncio
	@patch('app.features.models.api.model_manager')
	async def test_load_model_cancelled(self, mock_model_manager):
		"""Test model loading cancelled (returns 204 No Content)."""
		# Arrange
		mock_model_manager.load_model_async = AsyncMock(side_effect=CancellationException('Cancelled'))

		# Act
		result = await load_model(self.request)

		# Assert
		assert isinstance(result, Response)
		assert result.status_code == 204


class TestUnloadModelEndpoint:
	"""Test unload model endpoint."""

	@pytest.mark.asyncio
	@patch('app.features.models.api.model_manager')
	async def test_unload_model_success(self, mock_model_manager):
		"""Test successful model unloading."""
		# Arrange
		mock_model_manager.unload_model_async = AsyncMock()

		# Act
		result = await unload_model()

		# Assert
		mock_model_manager.unload_model_async.assert_called_once()
		assert isinstance(result, JSONResponseMessage)
		assert 'Model unloaded successfully' in result.body.decode()

	@pytest.mark.asyncio
	@patch('app.features.models.api.model_manager')
	async def test_unload_model_error(self, mock_model_manager):
		"""Test error handling during model unloading."""
		# Arrange
		mock_model_manager.unload_model_async = AsyncMock(side_effect=Exception('Failed to unload model'))

		# Act & Assert
		with pytest.raises(HTTPException) as exc_info:
			await unload_model()

		assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
		assert 'Failed to unload model' in str(exc_info.value.detail)


class TestModelRecommendationsEndpoint:
	"""Test model recommendations endpoint."""

	def setup_method(self):
		"""Setup test method."""
		self.db_mock = MagicMock(spec=Session)

	@patch('app.features.models.api.ModelRecommendationService')
	def test_get_model_recommendations_success(self, mock_recommendation_service_class):
		"""Test successful model recommendations retrieval."""
		# Arrange
		mock_recommendation_service = MagicMock()
		mock_recommendation_service_class.return_value = mock_recommendation_service
		mock_recommendations = MagicMock()
		mock_recommendations.sections = ['section1', 'section2']
		mock_recommendation_service.get_recommendations.return_value = mock_recommendations

		# Act
		result = get_model_recommendations(self.db_mock)

		# Assert
		mock_recommendation_service_class.assert_called_once_with(self.db_mock)
		mock_recommendation_service.get_recommendations.assert_called_once()
		assert result == mock_recommendations

	@patch('app.features.models.api.ModelRecommendationService')
	def test_get_model_recommendations_error(self, mock_recommendation_service_class):
		"""Test error handling during model recommendations retrieval."""
		# Arrange
		mock_recommendation_service = MagicMock()
		mock_recommendation_service_class.return_value = mock_recommendation_service
		mock_recommendation_service.get_recommendations.side_effect = Exception('Failed to generate recommendations')

		# Act & Assert
		with pytest.raises(HTTPException) as exc_info:
			get_model_recommendations(self.db_mock)

		assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
		assert 'Failed to generate model recommendations' in str(exc_info.value.detail)


class TestModelAvailabilityEndpoint:
	"""Test model availability endpoint."""

	def setup_method(self):
		"""Setup test method."""
		self.db_mock = MagicMock(spec=Session)
		self.model_id = 'test/model'

	@patch('app.features.models.api.model_service')
	def test_is_model_available_success(self, mock_model_service):
		"""Test successful model availability check."""
		# Arrange
		mock_model_service.is_model_downloaded.return_value = True

		# Act
		result = is_model_available(self.model_id, self.db_mock)

		# Assert
		mock_model_service.is_model_downloaded.assert_called_once_with(self.db_mock, self.model_id)
		assert isinstance(result, ModelAvailableResponse)
		assert result.id == self.model_id
		assert result.is_downloaded is True

	@patch('app.features.models.api.model_service')
	def test_is_model_available_error(self, mock_model_service):
		"""Test error handling during model availability check."""
		# Arrange
		error_message = 'Database error'
		mock_model_service.is_model_downloaded.side_effect = Exception(error_message)

		# Act & Assert
		with pytest.raises(HTTPException) as exc_info:
			is_model_available(self.model_id, self.db_mock)

		assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
		assert exc_info.value.detail == error_message


class TestListModelsEndpoint:
	"""Test list_models endpoint."""

	@patch('app.features.models.api.api')
	def test_list_models_success(self, mock_api):
		"""Test successful model listing from HuggingFace (lines 45-61)."""
		# Arrange
		mock_model1 = MagicMock()
		mock_model1.__dict__ = {
			'id': 'model1',
			'author': 'user1',
			'downloads': 100,
			'likes': 50,
			'tags': ['diffusion'],
		}
		mock_model2 = MagicMock()
		mock_model2.__dict__ = {
			'id': 'model2',
			'author': 'user2',
			'downloads': 200,
			'likes': 75,
			'tags': ['text-to-image'],
		}

		mock_api.list_models.return_value = iter([mock_model1, mock_model2])

		# Act
		from app.features.models.api import list_models

		result = list_models(filter='diffusion', limit=20, model_name=None, sort='likes')

		# Assert
		mock_api.list_models.assert_called_once_with(
			full=True,
			filter='diffusion',
			limit=20,
			model_name=None,
			pipeline_tag='text-to-image',
			sort='likes',
		)
		assert len(result.models_search_info) == 2
		assert result.models_search_info[0].id == 'model1'
		assert result.models_search_info[1].id == 'model2'


class TestGetModelInfoEndpoint:
	"""Test get_model_info endpoint."""

	@patch('app.features.models.api.api')
	def test_get_model_info_success(self, mock_api):
		"""Test successful model info retrieval (lines 67-75)."""
		# Arrange
		mock_model_info = {'id': 'test/model', 'author': 'test', 'pipeline_tag': 'text-to-image'}
		mock_api.model_info.return_value = mock_model_info

		# Act
		from app.features.models.api import get_model_info

		result = get_model_info(id='test/model')

		# Assert
		mock_api.model_info.assert_called_once_with('test/model', files_metadata=True)
		assert result == mock_model_info

	def test_get_model_info_missing_id(self):
		"""Test error when ID is missing or empty."""
		from app.features.models.api import get_model_info

		# Test with empty string
		with pytest.raises(HTTPException) as exc_info:
			get_model_info(id='')

		assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
		assert "Missing 'id' query parameter" in str(exc_info.value.detail)


class TestGetDownloadedModelsEndpoint:
	"""Test get_downloaded_models endpoint."""

	@patch('app.features.models.api.model_service')
	def test_get_downloaded_models_success(self, mock_model_service):
		"""Test successful downloaded models retrieval (lines 81-88)."""
		# Arrange
		db_mock = MagicMock(spec=Session)
		mock_models = [{'id': 'model1'}, {'id': 'model2'}]
		mock_model_service.get_downloaded_models.return_value = mock_models

		# Act
		from app.features.models.api import get_downloaded_models

		result = get_downloaded_models(db=db_mock)

		# Assert
		mock_model_service.get_downloaded_models.assert_called_once_with(db_mock)
		assert result == mock_models

	@patch('app.features.models.api.model_service')
	def test_get_downloaded_models_error(self, mock_model_service):
		"""Test error handling in get_downloaded_models."""
		# Arrange
		db_mock = MagicMock(spec=Session)
		mock_model_service.get_downloaded_models.side_effect = Exception('Database error')

		# Act & Assert
		from app.features.models.api import get_downloaded_models

		with pytest.raises(HTTPException) as exc_info:
			get_downloaded_models(db=db_mock)

		assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestGetModelStatusEndpoint:
	"""Test get_model_status endpoint."""

	@patch('app.features.models.api.model_manager')
	def test_get_model_status_success(self, mock_model_manager):
		"""Test successful model status retrieval (lines 200-216)."""
		# Arrange
		from app.cores.model_manager import ModelState

		mock_model_manager.get_state.return_value = ModelState.LOADED
		mock_model_manager.id = 'test/model'
		mock_model_manager.pipe = MagicMock()

		# Act
		from app.features.models.api import get_model_status

		result = get_model_status()

		# Assert
		assert result['state'] == 'loaded'
		assert result['loaded_model_id'] == 'test/model'
		assert result['has_model'] is True
		assert result['is_loading'] is False
		assert result['is_cancelling'] is False

	@patch('app.features.models.api.model_manager')
	def test_get_model_status_loading_state(self, mock_model_manager):
		"""Test model status when loading."""
		# Arrange
		from app.cores.model_manager import ModelState

		mock_model_manager.get_state.return_value = ModelState.LOADING
		mock_model_manager.id = None
		mock_model_manager.pipe = None

		# Act
		from app.features.models.api import get_model_status

		result = get_model_status()

		# Assert
		assert result['state'] == 'loading'
		assert result['is_loading'] is True
		assert result['has_model'] is False

	@patch('app.features.models.api.model_manager')
	def test_get_model_status_error(self, mock_model_manager):
		"""Test error handling in get_model_status."""
		# Arrange
		mock_model_manager.get_state.side_effect = Exception('Failed to get state')

		# Act & Assert
		from app.features.models.api import get_model_status

		with pytest.raises(HTTPException) as exc_info:
			get_model_status()

		assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
		assert 'Failed to get model status' in str(exc_info.value.detail)
