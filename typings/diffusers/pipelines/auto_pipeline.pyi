"""Type stubs for diffusers.pipelines.auto_pipeline module."""

from collections.abc import Mapping
from typing import Any

import torch

class SchedulerConfig:
	"""Scheduler configuration."""

	def __getattr__(self, name: str) -> Any: ...

class UNetConfig:
	"""UNet configuration."""

	sample_size: int
	def __getattr__(self, name: str) -> Any: ...

class Scheduler:
	"""Base scheduler class."""

	config: SchedulerConfig
	@classmethod
	def from_config(cls, config: SchedulerConfig, **kwargs: Any) -> Scheduler: ...

class UNet:
	"""UNet model."""

	config: UNetConfig

class AutoPipelineForText2Image:
	"""Auto pipeline for text-to-image generation."""

	scheduler: Scheduler
	unet: UNet
	device: torch.device
	config: Mapping[str, object]

	def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
	def load_lora_weights(self, pretrained_model_name_or_path_or_dict: str, **kwargs: Any) -> None: ...
	def set_adapters(self, adapter_names: list[str], adapter_weights: list[float] | None = None) -> None: ...
	def unload_lora_weights(self) -> None: ...
	@classmethod
	def from_pipe(cls, pipe: Any, **kwargs: Any) -> AutoPipelineForText2Image: ...
	@classmethod
	def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: Any) -> AutoPipelineForText2Image: ...

class AutoPipelineForImage2Image:
	"""Auto pipeline for image-to-image generation."""

	scheduler: Scheduler
	unet: UNet
	device: torch.device
	config: Mapping[str, object]

	def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
	def load_lora_weights(self, pretrained_model_name_or_path_or_dict: str, **kwargs: Any) -> None: ...
	def set_adapters(self, adapter_names: list[str], adapter_weights: list[float] | None = None) -> None: ...
	def unload_lora_weights(self) -> None: ...
	@classmethod
	def from_pipe(cls, pipe: AutoPipelineForText2Image, **kwargs: Any) -> AutoPipelineForImage2Image: ...
	@classmethod
	def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: Any) -> AutoPipelineForImage2Image: ...
