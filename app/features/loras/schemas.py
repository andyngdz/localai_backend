"""LoRA schemas for API requests and responses."""

from datetime import datetime

from pydantic import BaseModel, Field


class LoRAUploadRequest(BaseModel):
	"""Request model for uploading a LoRA file."""

	file_path: str = Field(..., description='Path to the LoRA file on the local filesystem.')


class LoRAConfigItem(BaseModel):
	"""Configuration for applying a LoRA during generation."""

	lora_id: int = Field(..., description='Database ID of the LoRA to apply.')
	weight: float = Field(default=1.0, ge=0.0, le=2.0, description='Weight/strength of the LoRA (0.0-2.0).')


class LoRAInfo(BaseModel):
	"""Response model for LoRA information."""

	id: int = Field(..., description='Database ID of the LoRA.')
	name: str = Field(..., description='Display name of the LoRA.')
	file_path: str = Field(..., description='Path to the LoRA file.')
	file_size: int = Field(..., description='Size of the LoRA file in bytes.')
	created_at: datetime = Field(..., description='When the LoRA was added.')
	updated_at: datetime = Field(..., description='When the LoRA was last updated.')


class LoRAListResponse(BaseModel):
	"""Response model for listing LoRAs."""

	loras: list[LoRAInfo] = Field(default_factory=list, description='List of available LoRAs.')


class LoRADeleteResponse(BaseModel):
	"""Response model for deleting a LoRA."""

	id: int = Field(..., description='ID of the deleted LoRA.')
	message: str = Field(..., description='Confirmation message.')
