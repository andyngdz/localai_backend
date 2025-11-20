"""Type stubs for transformers.tokenization_utils_base module."""

from typing import Any

class BatchEncoding:
	"""Dict-like object returned by tokenizers."""

	input_ids: list[int]
	attention_mask: list[int] | None

	def __getitem__(self, key: str) -> Any:
		"""Get item by key (dict-like access)."""
		...

	def keys(self) -> Any:
		"""Get keys (dict-like behavior)."""
		...
