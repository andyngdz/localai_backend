"""Unit tests for app/routers/downloads/schemas.py

Covers basic validation rules for Pydantic models:
- Required vs optional fields
- Allowing None for Optional[...] with required Field(...)
- Type coercion for numeric fields (Pydantic v2 behavior)
- Minimal serialization with model_dump()
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.features.downloads.schemas import (
	DownloadModelRequest,
	DownloadModelResponse,
	DownloadModelStartResponse,
	DownloadStepProgressResponse,
)

# DownloadModelRequest


def test_download_model_request_accepts_id_and_defaults_hf_token_to_none() -> None:
	# Arrange & Act
	model = DownloadModelRequest(id='org/model')

	# Assert
	assert model.id == 'org/model'
	assert model.hf_token is None


def test_download_model_request_requires_id() -> None:
	# Arrange & Act & Assert
	with pytest.raises(ValidationError):
		DownloadModelRequest()  # type: ignore[call-arg]


def test_download_model_request_preserves_hf_token_when_provided() -> None:
	# Arrange & Act
	model = DownloadModelRequest(id='org/model', hf_token='hf_abc123')

	# Assert
	assert model.hf_token == 'hf_abc123'


# DownloadModelStartResponse


def test_download_model_start_response_requires_id() -> None:
	# Arrange & Act & Assert
	with pytest.raises(ValidationError):
		DownloadModelStartResponse()  # type: ignore[call-arg]


# DownloadStepProgressResponse


def test_download_step_progress_response_requires_all_fields() -> None:
	# Arrange & Act & Assert
	with pytest.raises(ValidationError):
		DownloadStepProgressResponse(id='org/model')  # type: ignore[call-arg]


def test_download_step_progress_response_coerces_int_fields_from_str() -> None:
	# Arrange & Act
	model = DownloadStepProgressResponse(id='org/model', step='1', total='3')  # type: ignore[arg-type]

	# Assert
	assert model.step == 1
	assert model.total == 3


# DownloadModelResponse


def test_download_model_response_requires_message_field_even_if_optional_type() -> None:
	# Arrange & Act & Assert
	# message is Optional[str] but Field(...) => required key
	with pytest.raises(ValidationError):
		DownloadModelResponse(id='org/model')  # type: ignore[call-arg]


def test_download_model_response_allows_message_none() -> None:
	# Arrange & Act
	model = DownloadModelResponse(id='org/model', message=None)

	# Assert
	assert model.id == 'org/model'
	assert model.message is None


def test_download_model_response_accepts_message_string_and_serializes() -> None:
	# Arrange
	model = DownloadModelResponse(id='org/model', message='Download completed')

	# Act
	payload = model.model_dump()

	# Assert
	assert payload == {'id': 'org/model', 'message': 'Download completed'}
