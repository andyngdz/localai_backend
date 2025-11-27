import torch
from PIL import Image
from torch import nn

class StableDiffusionSafetyChecker(nn.Module):
	def __call__(
		self,
		images: list[Image.Image],
		clip_input: torch.Tensor,
	) -> tuple[list[Image.Image], list[bool]]: ...
	
	@classmethod
	def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> 'StableDiffusionSafetyChecker': ...
