from enum import Enum
from typing import Literal, Optional, Protocol, TypeAlias, Union

import torch
from diffusers.pipelines.auto_pipeline import AutoPipelineForImage2Image, AutoPipelineForText2Image
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline
from pydantic import BaseModel, Field


class ModelLoadPhase(str, Enum):
	"""
	Enum for the different phases of model loading.
	Using str as base class ensures it serializes to string in JSON.
	"""

	INITIALIZATION = 'initialization'
	LOADING_MODEL = 'loading_model'
	DEVICE_SETUP = 'device_setup'
	OPTIMIZATION = 'optimization'


class ModelLoadProgressResponse(BaseModel):
	"""Response model for model loading progress updates."""

	model_id: str = Field(..., description='The ID of the model being loaded.')
	step: int = Field(..., description='Current checkpoint (1-9).')
	total: int = Field(default=9, description='Total checkpoints.')
	phase: ModelLoadPhase = Field(..., description='Current loading phase.')
	message: str = Field(..., description='Human-readable progress message.')


class ModelLoadCompletedResponse(BaseModel):
	"""Response model for when a model has been successfully loaded."""

	model_id: str = Field(..., description='The ID of the model that was loaded.')


class ModelLoadFailed(BaseModel):
	"""Response model for when a model has failed to load."""

	model_id: str = Field(..., description='The ID of the model that failed to load.')
	error: str = Field(..., description='The error message.')


class ModelLoaderProgressStep(BaseModel):
	"""Progress step for model loader initialization."""

	id: int = Field(..., description='Step number.')
	message: str = Field(..., description='Progress message for this step.')


class SingleFileStrategy(BaseModel):
	"""Strategy for loading models from a single checkpoint file."""

	checkpoint_path: str
	type: Literal['single_file'] = 'single_file'


class PretrainedStrategy(BaseModel):
	"""Strategy for loading pretrained models from HuggingFace Hub."""

	use_safetensors: bool
	variant: Optional[str] = None
	type: Literal['pretrained'] = 'pretrained'


Strategy = Union[SingleFileStrategy, PretrainedStrategy]


class DiffusersPipelineProtocol(Protocol):
	"""Protocol defining common interface for all diffusers pipelines."""

	device: torch.device
	dtype: torch.dtype

	def to(self, device: Optional[Union[str, torch.device]] = None) -> 'DiffusersPipelineProtocol': ...
	def to_empty(self, device: Optional[Union[str, torch.device]] = None) -> 'DiffusersPipelineProtocol': ...
	def __call__(self, **kwargs): ...


# Type alias for diffusers pipelines - includes both auto pipelines and specific implementations
DiffusersPipeline: TypeAlias = (
	AutoPipelineForText2Image
	| AutoPipelineForImage2Image
	| StableDiffusionXLPipeline
	| StableDiffusionPipeline
	| StableDiffusion3Pipeline
)

# Type alias for pipeline classes that support from_single_file
SingleFilePipelineClass: TypeAlias = (
	type[StableDiffusionXLPipeline] | type[StableDiffusionPipeline] | type[StableDiffusion3Pipeline]
)
