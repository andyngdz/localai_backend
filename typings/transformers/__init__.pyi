"""Type stubs for transformers library.

This module provides type hints for the transformers library APIs used in this project.
"""

from typing import Any

from .tokenization_utils_base import BatchEncoding

class CLIPImageProcessor:
	"""CLIP image processor."""

	@classmethod
	def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: Any) -> CLIPImageProcessor:
		"""Load pre-trained image processor."""
		...

class CLIPTokenizer:
	"""CLIP tokenizer for encoding text."""

	@classmethod
	def from_pretrained(
		cls,
		pretrained_model_name_or_path: str,
		*,
		local_files_only: bool = False,
		**kwargs: Any,
	) -> CLIPTokenizer:
		"""Load pre-trained tokenizer."""
		...

	def __call__(
		self,
		text: str,
		truncation: bool = False,
		max_length: int = 77,
		**kwargs: Any,
	) -> BatchEncoding:
		"""Tokenize text and return BatchEncoding."""
		...

class CLIPTextModel:
	"""CLIP text encoder model."""

	@classmethod
	def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: Any) -> 'CLIPTextModel':
		"""Load pre-trained text encoder."""
		...

class CLIPTextModelWithProjection:
	"""CLIP text encoder model with projection layer (used in SDXL)."""

	@classmethod
	def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: Any) -> 'CLIPTextModelWithProjection':
		"""Load pre-trained text encoder with projection."""
		...

class GPT2TokenizerFast:
	"""GPT-2 fast tokenizer."""

	@classmethod
	def from_pretrained(
		cls,
		pretrained_model_name_or_path: str,
		**kwargs: Any,
	) -> GPT2TokenizerFast:
		"""Load pre-trained tokenizer."""
		...

	def encode(self, text: str, **kwargs: Any) -> list[int]:
		"""Encode text to token IDs."""
		...

	def decode(self, token_ids: list[int], **kwargs: Any) -> str:
		"""Decode token IDs to text."""
		...
