from typing import Any

class StableDiffusionSafetyChecker:
	@classmethod
	def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: Any) -> StableDiffusionSafetyChecker: ...
