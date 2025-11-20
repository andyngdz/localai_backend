"""Type stubs for diffusers.pipelines.auto_pipeline module."""

from collections.abc import Mapping
from typing import Any, Optional, Union

import torch
from transformers import CLIPImageProcessor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

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
	safety_checker: Optional[StableDiffusionSafetyChecker]
	feature_extractor: Optional[CLIPImageProcessor]

	def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
	def load_lora_weights(self, pretrained_model_name_or_path_or_dict: str, **kwargs: Any) -> None: ...
	def set_adapters(self, adapter_names: list[str], adapter_weights: Optional[list[float]] = None) -> None: ...
	def unload_lora_weights(self) -> None: ...
	def reset_device_map(self) -> None: ...
	def to(
		self,
		device: Optional[Union[str, torch.device]] = None,
		dtype: Optional[torch.dtype] = None,
	) -> AutoPipelineForText2Image: ...
	def to_empty(
		self,
		device: Optional[Union[str, torch.device]] = None,
	) -> AutoPipelineForText2Image: ...
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
	safety_checker: Optional[StableDiffusionSafetyChecker]
	feature_extractor: Optional[CLIPImageProcessor]

	def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
	def load_lora_weights(self, pretrained_model_name_or_path_or_dict: str, **kwargs: Any) -> None: ...
	def set_adapters(self, adapter_names: list[str], adapter_weights: Optional[list[float]] = None) -> None: ...
	def unload_lora_weights(self) -> None: ...
	def reset_device_map(self) -> None: ...
	def to(
		self,
		device: Optional[Union[str, torch.device]] = None,
		dtype: Optional[torch.dtype] = None,
	) -> AutoPipelineForImage2Image: ...
	def to_empty(
		self,
		device: Optional[Union[str, torch.device]] = None,
	) -> AutoPipelineForImage2Image: ...
	@classmethod
	def from_pipe(cls, pipe: AutoPipelineForText2Image, **kwargs: Any) -> AutoPipelineForImage2Image: ...
	@classmethod
	def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: Any) -> AutoPipelineForImage2Image: ...
