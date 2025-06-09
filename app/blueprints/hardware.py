"""Blueprint for handling drivers in the application."""

import functools
import logging
import platform
import subprocess

import torch
from flask import Blueprint, jsonify, request
from pydantic import ValidationError

from app.database.database import create_or_update_selected_device
from app.schemas.core import ErrorResponse, ErrorType
from app.schemas.hardware import (
    GPUDeviceInfo,
    GPUDriverInfo,
    GPUDriverStatusStates,
    SelectDeviceRequest,
)

logger = logging.getLogger(__name__)

hardware = Blueprint("hardware", __name__)


def _create_initial_gpu_info() -> GPUDriverInfo:
    """Initializes GPUDriverInfo with default unknown status."""
    return GPUDriverInfo(
        overall_status=GPUDriverStatusStates.UNKNOWN_ERROR,
        message="Attempting to detect GPU and driver status...",
        detected_gpus=[],
        nvidia_driver_version=None,
        cuda_runtime_version=None,
        macos_mps_available=None,
        recommendation_link=None,
        troubleshooting_steps=None,
    )


def _get_nvidia_gpu_info(system_os: str, info: GPUDriverInfo):
    # NVIDIA GPU Detection
    cuda = torch.cuda

    if cuda.is_available():
        info.overall_status = GPUDriverStatusStates.READY
        info.message = "NVIDIA GPU detected and ready for acceleration."
        version = getattr(torch, "version", None)
        info.cuda_runtime_version = getattr(version, "cuda", None)

        # Get individual GPU device info
        detected_gpus = []
        for i in range(cuda.device_count()):
            device_name = cuda.get_device_name(i)
            device_property = cuda.get_device_properties(i)
            total_memory = device_property.total_memory / (1024**2)  # Convert to MB
            compute_capability = f"{device_property.major}.{device_property.minor}"
            detected_gpus.append(
                GPUDeviceInfo(
                    name=device_name,
                    memory_mb=int(total_memory),
                    cuda_compute_capability=compute_capability,
                    is_primary=(i == cuda.current_device()),
                )
            )
        info.detected_gpus = detected_gpus

        # Try to get NVIDIA driver version via nvidia-smi
        try:
            # Capture stdout and stderr
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                check=True,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW if system_os == "Windows" else 0
                ),
            )
            driver_version = result.stdout.strip().split("\n")[0]
            info.nvidia_driver_version = driver_version
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(
                "nvidia-smi not found or failed: %s. Cannot get detailed driver version.",
                e,
            )
            info.message += " (Could not retrieve NVIDIA driver version from nvidia-smi. Ensure it's in PATH)."
            # If CUDA is available but nvidia-smi fails, it's still "ready" but with a warning.
            # If it's a driver issue, torch.cuda.is_available() would likely be False.
    else:
        # No NVIDIA GPU or CUDA issues
        info.overall_status = GPUDriverStatusStates.NO_GPU
        info.message = (
            "No NVIDIA GPU detected or CUDA is not available. Running on CPU."
        )
        info.recommendation_link = (
            "https://www.nvidia.com/drivers"  # Generic NVIDIA driver link
        )
        info.troubleshooting_steps = [
            "Ensure you have an NVIDIA GPU.",
            "Download and install the latest NVIDIA drivers for your system.",
            "Verify PyTorch is installed with CUDA support (e.g., pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118).",
        ]
        # Check if nvidia-smi exists but reports no GPUs or errors
        try:
            subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                check=True,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW if system_os == "Windows" else 0
                ),
            )
            # If nvidia-smi runs but torch.cuda.is_available() is false, it's likely a driver/CUDA setup issue
            info.overall_status = GPUDriverStatusStates.DRIVER_ISSUE
            info.message = "NVIDIA GPU detected, but CUDA is not available or drivers are incompatible. Please update your drivers."
            info.troubleshooting_steps.insert(
                0, "Check NVIDIA Control Panel/Settings for driver status."
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            # nvidia-smi also not found, so likely no NVIDIA GPU or severe path issue
            pass


def _get_mps_gpu_info(info: GPUDriverInfo):
    info.overall_status = GPUDriverStatusStates.READY
    info.message = "Apple Silicon (MPS) GPU detected and ready for acceleration."
    info.macos_mps_available = True
    # MPS doesn't expose detailed device info like CUDA, but you can get general info
    # For simplicity, we'll just add a generic entry if MPS is available
    info.detected_gpus.append(
        GPUDeviceInfo(
            name=platform.machine(),  # e.g., 'arm64' for M-series
            memory_mb=None,  # MPS doesn't expose dedicated VRAM easily
            cuda_compute_capability=None,
            is_primary=True,
        )
    )


@functools.cache
def _get_system_gpu_info() -> GPUDriverInfo:
    """
    Detects GPU and driver information based on the operating system.
    This function will be called by the endpoints.
    """

    # Initialize with default unknown status
    info = _create_initial_gpu_info()

    try:
        system_os = platform.system()
        backends = torch.backends

        if system_os in ("Windows", "Linux"):
            _get_nvidia_gpu_info(system_os, info)
        elif system_os == "Darwin":  # macOS
            # Apple Silicon (MPS) Detection
            if hasattr(backends, "mps") and backends.mps.is_available():
                _get_mps_gpu_info(info)
            else:
                info.overall_status = GPUDriverStatusStates.NO_GPU
                info.message = "No Apple Silicon (MPS) GPU detected or PyTorch MPS backend not available. Running on CPU."
                info.macos_mps_available = False
                info.troubleshooting_steps = [
                    "Ensure you are running on an Apple Silicon Mac (M1, M2, etc.).",
                    "Update macOS to the latest version.",
                    "Verify PyTorch is installed with MPS support (e.g., pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu for CPU or specific MPS build).",
                ]
        else:
            # Other OS / Unknown
            info.overall_status = GPUDriverStatusStates.NO_GPU
            info.message = f"Unsupported operating system '{system_os}' or no compatible GPU detected. Running on CPU."

    except ImportError:
        info.overall_status = GPUDriverStatusStates.UNKNOWN_ERROR
        info.message = "PyTorch is not installed or not accessible in the backend environment. Cannot detect GPU."
        info.troubleshooting_steps = [
            "Ensure PyTorch is correctly installed in the backend's Python environment."
        ]
    except (subprocess.SubprocessError, OSError, AttributeError, RuntimeError) as e:
        logger.error("Error during GPU detection: %s", exc_info=True)
        info.overall_status = GPUDriverStatusStates.UNKNOWN_ERROR
        info.message = f"An unexpected error occurred during GPU detection: {str(e)}. Running on CPU."
        info.troubleshooting_steps = [
            "Check backend logs for more details.",
            "Ensure all dependencies are correctly installed.",
        ]

    return info


@hardware.route("/status", methods=["GET"])
def get_driver_status():
    """
    Returns the current GPU and driver status of the system.
    This will return the cached result of _get_system_gpu_info().
    """
    driver_info = _get_system_gpu_info()

    return jsonify(driver_info.model_dump()), 200


@hardware.route("/recheck", methods=["GET"])
def recheck_driver_status():
    """
    Forces the backend to re-evaluate and update the GPU and driver status.
    This clears the cache of _get_system_gpu_info() and then calls it again.
    """
    _get_system_gpu_info.cache_clear()  # Clear the cache

    logger.info("Forcing re-check of GPU driver status by clearing cache.")
    driver_info = _get_system_gpu_info()  # Re-run detection (will now actually execute)

    return jsonify(driver_info.model_dump()), 200


@hardware.route("/device", methods=["POST"])
def select_device():
    """Select device"""
    try:
        json = request.get_json()
        data = SelectDeviceRequest(**json)
    except ValidationError as e:
        return jsonify(
            ErrorResponse(
                detail=e.errors(), type=ErrorType.ValidationError
            ).model_dump()
        ), 400

    device_index = data.device_index
    create_or_update_selected_device(device_index=device_index)

    return jsonify({"message": "Device selected successfully"}), 200
