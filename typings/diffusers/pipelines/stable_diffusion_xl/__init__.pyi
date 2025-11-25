"""Type stubs for diffusers SDXL pipelines."""

from collections.abc import Mapping
from typing import Any, Optional, Union

import torch
from diffusers.pipelines.auto_pipeline import Scheduler, UNet
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

class StableDiffusionXLPipeline:
	"""Stable Diffusion XL pipeline."""

	# Core components
	scheduler: Scheduler
	unet: UNet
	device: torch.device
	config: Mapping[str, object]
	safety_checker: Optional[StableDiffusionSafetyChecker]
	feature_extractor: Optional[CLIPImageProcessor]

	# SDXL-specific: dual text encoders and tokenizers
	tokenizer: CLIPTokenizer
	tokenizer_2: Optional[CLIPTokenizer]
	text_encoder: CLIPTextModel
	text_encoder_2: Optional[CLIPTextModelWithProjection]

	@classmethod
	def from_single_file(cls, pretrained_model_link_or_path: str, **kwargs: Any) -> 'StableDiffusionXLPipeline': ...
	def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
	def load_lora_weights(self, pretrained_model_name_or_path_or_dict: str, **kwargs: Any) -> None: ...
	def set_adapters(self, adapter_names: list[str], adapter_weights: Optional[list[float]] = None) -> None: ...
	def unload_lora_weights(self) -> None: ...
	def reset_device_map(self) -> None: ...
	def to(
		self,
		device: Optional[Union[str, torch.device]] = None,
		dtype: Optional[torch.dtype] = None,
	) -> 'StableDiffusionXLPipeline': ...
	def to_empty(
		self,
		device: Optional[Union[str, torch.device]] = None,
	) -> 'StableDiffusionXLPipeline': ...
