"""Schemas for GPU/driver detection and status reporting."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class GPUDriverStatusStates(str, Enum):
	"""Overall status of GPU/driver setup for AI acceleration."""

	READY = 'ready'
	NO_GPU = 'no_gpu'
	DRIVER_ISSUE = 'driver_issue'
	INCOMPATIBLE_CUDA = 'incompatible_cuda'
	UNKNOWN_ERROR = 'unknown_error'


class GPUDeviceInfo(BaseModel):
	"""Information about a single detected GPU device."""

	name: str = Field(
		...,
		description="Name of the GPU (e.g., 'NVIDIA GeForce RTX 3080', 'Apple M1 Max').",
	)
	memory: Optional[int] = Field(default=0, description='Total memory of the GPU.')
	cuda_compute_capability: Optional[str] = Field(
		default='0', description="CUDA compute capability (e.g., '8.6') if NVIDIA GPU."
	)
	is_primary: bool = Field(default=False, description='True if this is the primary/default GPU.')


class GPUDriverInfo(BaseModel):
	"""Comprehensive information about the system's GPU and driver setup."""

	is_cuda: bool = Field(default=False, description='True if CUDA is available and NVIDIA GPU is detected.')
	overall_status: GPUDriverStatusStates = Field(..., description='Overall status of GPU/driver setup.')
	message: str = Field(..., description='A user-friendly message explaining the status.')
	gpus: List[GPUDeviceInfo] = Field(default_factory=list, description='List of detected GPU devices.')
	nvidia_driver_version: Optional[str] = Field(
		default='0', description='NVIDIA driver version (if NVIDIA GPU detected).'
	)
	cuda_runtime_version: Optional[str] = Field(
		default='0',
		description='CUDA runtime version detected by PyTorch/system (if NVIDIA GPU).',
	)
	macos_mps_available: Optional[bool] = Field(
		default=False,
		description='True if Metal Performance Shaders (MPS) are available on macOS.',
	)
	recommendation_link: Optional[str] = Field(
		default='',
		description='A URL for recommended driver downloads or troubleshooting.',
	)
	troubleshooting_steps: Optional[List[str]] = Field(
		default_factory=lambda: [], description='Specific steps to resolve issues.'
	)


class SelectDeviceRequest(BaseModel):
	"""Request for selecting a device with index"""

	device_index: int = Field(
		...,
		ge=0,
		description='Index of the device to select. -2 means not found, -1 means CPU mode.',
	)


class GetCurrentDeviceIndex(BaseModel):
	"""Request to get the current selected device index"""

	device_index: int = Field(
		...,
		description='Index of the currently selected device. -2 means not found, -1 means CPU mode.',
	)


class MaxMemoryConfigRequest(BaseModel):
	"""Configuration for maximum memory usage."""

	ram_scale_factor: float = Field(
		...,
		ge=0.1,
		le=1,
		description='Maximum RAM memory in percent that can be used by the pipeline.',
	)
	gpu_scale_factor: float = Field(
		...,
		ge=0.1,
		le=1,
		description='Maximum GPU memory in percent that can be used by the pipeline.',
	)


class MemoryResponse(BaseModel):
	"""Response containing the maximum memory configuration."""

	gpu: int = Field(
		...,
		description='Maximum GPU memory in bytes.',
	)
	ram: int = Field(
		...,
		description='Maximum RAM memory in bytes.',
	)
