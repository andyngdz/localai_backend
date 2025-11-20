from typing import Any

from torch import nn


class StableDiffusionSafetyChecker(nn.Module):
	@classmethod
	def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: Any) -> StableDiffusionSafetyChecker: ...
