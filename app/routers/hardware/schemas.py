"""Schemas for GPU/driver detection and status reporting."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class GPUDriverStatusStates(str, Enum):
    """Overall status of GPU/driver setup for AI acceleration."""

    READY = 'ready'  # GPU detected, drivers compatible, ready for AI acceleration
    NO_GPU = 'no_gpu'  # No compatible GPU detected
    DRIVER_ISSUE = (
        'driver_issue'  # GPU detected, but drivers are missing/outdated/problematic
    )
    INCOMPATIBLE_CUDA = 'incompatible_cuda'  # GPU/drivers fine, but CUDA version incompatible with backend's PyTorch
    UNKNOWN_ERROR = 'unknown_error'  # An unexpected error occurred during detection


class GPUDeviceInfo(BaseModel):
    """Information about a single detected GPU device."""

    name: str = Field(
        ...,
        description="Name of the GPU (e.g., 'NVIDIA GeForce RTX 3080', 'Apple M1 Max').",
    )
    memory_mb: Optional[int] = Field(None, description='Total memory of the GPU in MB.')
    cuda_compute_capability: Optional[str] = Field(
        None, description="CUDA compute capability (e.g., '8.6') if NVIDIA GPU."
    )
    is_primary: bool = Field(
        False, description='True if this is the primary/default GPU.'
    )


class GPUDriverInfo(BaseModel):
    """Comprehensive information about the system's GPU and driver setup."""

    overall_status: GPUDriverStatusStates = Field(
        ..., description='Overall status of GPU/driver setup.'
    )
    message: str = Field(
        ..., description='A user-friendly message explaining the status.'
    )
    detected_gpus: List[GPUDeviceInfo] = Field(
        default_factory=list, description='List of detected GPU devices.'
    )
    nvidia_driver_version: Optional[str] = Field(
        default=None, description='NVIDIA driver version (if NVIDIA GPU detected).'
    )
    cuda_runtime_version: Optional[str] = Field(
        default=None,
        description='CUDA runtime version detected by PyTorch/system (if NVIDIA GPU).',
    )
    # Specifics for Apple Silicon
    macos_mps_available: Optional[bool] = Field(
        default=None,
        description='True if Metal Performance Shaders (MPS) are available on macOS.',
    )
    # Add recommendations or links
    recommendation_link: Optional[str] = Field(
        default=None,
        description='A URL for recommended driver downloads or troubleshooting.',
    )
    troubleshooting_steps: Optional[List[str]] = Field(
        default=None, description='Specific steps to resolve issues.'
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

    ram: float = Field(
        ...,
        ge=0.1,
        le=1,
        description='Maximum RAM memory in percent that can be used by the pipeline.',
    )
    gpu: float = Field(
        ...,
        ge=0.1,
        le=1,
        description='Maximum GPU memory in percent that can be used by the pipeline.',
    )
