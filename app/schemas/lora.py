"""LoRA shared schemas."""

from pydantic import BaseModel, Field


class LoRAConfigItem(BaseModel):
	"""Configuration for applying a LoRA during generation."""

	lora_id: int = Field(..., description='Database ID of the LoRA to apply.')
	weight: float = Field(default=1.0, ge=0.0, le=2.0, description='Weight/strength of the LoRA (0.0-2.0).')


class LoRAData(BaseModel):
	"""Data structure for LoRA configuration passed to pipeline manager."""

	id: int = Field(..., description='Database ID of the LoRA.')
	name: str = Field(..., description='Display name of the LoRA.')
	file_path: str = Field(..., description='Path to the LoRA file.')
	weight: float = Field(..., ge=0.0, le=2.0, description='Weight/strength of the LoRA.')
