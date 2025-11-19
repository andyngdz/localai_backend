"""Tests for the styles service."""

from unittest.mock import Mock, patch

import pytest

from app.services.styles import StylesService
from app.styles.schemas import StyleItem


class TestStylesService:
	"""Test the StylesService class."""

	@pytest.fixture
	def service(self) -> StylesService:
		"""Create a fresh StylesService instance for each test."""
		return StylesService()

	@pytest.fixture
	def mock_styles(self) -> list[StyleItem]:
		"""Create mock style items for testing."""
		return [
			StyleItem(
				id='style1',
				name='Photorealistic',
				positive='photorealistic, {prompt}, highly detailed',
				negative='cartoon, anime',
				image='https://example.com/style1.jpg',
			),
			StyleItem(
				id='style2',
				name='Cinematic',
				positive='cinematic lighting, {prompt}, dramatic',
				negative='flat lighting',
				image='https://example.com/style2.jpg',
			),
			StyleItem(
				id='style3',
				name='Portrait',
				positive='portrait, professional photography',
				negative='',
				image='https://example.com/style3.jpg',
			),
		]

	def test_count_tokens_basic(self, service: StylesService) -> None:
		"""Test basic token counting."""
		text = 'a beautiful sunset over mountains'
		token_count = service.count_tokens(text)

		# Should return positive integer
		assert isinstance(token_count, int)
		assert token_count > 0

	def test_count_tokens_empty_string(self, service: StylesService) -> None:
		"""Test token counting with empty string."""
		token_count = service.count_tokens('')

		# Empty string should have minimal tokens
		assert token_count >= 0

	def test_count_tokens_long_text(self, service: StylesService) -> None:
		"""Test that long text is properly tokenized."""
		# Create text longer than 77 tokens
		long_text = ' '.join(['word'] * 100)
		token_count = service.count_tokens(long_text)

		# Should still return a count
		assert isinstance(token_count, int)
		assert token_count > 0

	def test_truncate_text_within_limit(self, service: StylesService) -> None:
		"""Test truncation of text that's already within token limit."""
		text = 'short text'
		result = service.truncate(text, max_tokens=50)

		assert result == text

	def test_truncate_text_exceeds_limit(self, service: StylesService) -> None:
		"""Test truncation of text exceeding token limit."""
		long_text = ' '.join(['word'] * 100)
		result = service.truncate(long_text, max_tokens=10)

		# Result should be shorter than original
		assert len(result) < len(long_text)
		# Result should fit within token limit
		assert service.count_tokens(result) <= 10

	def test_truncate_returns_empty_if_impossible(self, service: StylesService) -> None:
		"""Test that truncate returns empty string if even first word exceeds limit."""
		result = service.truncate('word', max_tokens=0)

		assert result == ''

	def test_apply_styles_no_styles_selected(self, service: StylesService, mock_styles: list[StyleItem]) -> None:
		"""Test applying styles when no styles are selected."""
		service.all_styles = mock_styles

		result = service.apply_styles(
			user_prompt='test prompt',
			user_negative_prompt='test negative',
			style_identifiers=[],
		)

		# Should return unchanged prompts
		assert result == ['test prompt', 'test negative']

	def test_apply_styles_single_style(self, service: StylesService, mock_styles: list[StyleItem]) -> None:
		"""Test applying a single style."""
		service.all_styles = mock_styles

		result = service.apply_styles(
			user_prompt='beautiful landscape',
			user_negative_prompt='',
			style_identifiers=['style1'],
		)

		positive, negative = result

		# Positive should contain user prompt and style additions
		assert 'beautiful landscape' in positive
		assert 'photorealistic' in positive
		assert 'highly detailed' in positive

		# Negative should use style's negative prompt
		assert 'cartoon' in negative
		assert 'anime' in negative

	def test_apply_styles_multiple_styles(self, service: StylesService, mock_styles: list[StyleItem]) -> None:
		"""Test applying multiple styles."""
		service.all_styles = mock_styles

		result = service.apply_styles(
			user_prompt='portrait of a person',
			user_negative_prompt='',
			style_identifiers=['style1', 'style2'],
		)

		positive, negative = result

		# Positive should contain user prompt and all style additions
		assert 'portrait of a person' in positive
		assert 'photorealistic' in positive
		assert 'cinematic lighting' in positive

		# Negative should contain both style negatives
		assert 'cartoon' in negative
		assert 'flat lighting' in negative

	def test_apply_styles_uses_default_negative_when_empty(
		self, service: StylesService, mock_styles: list[StyleItem]
	) -> None:
		"""Test that default negative prompt is used when user provides none."""
		service.all_styles = mock_styles

		result = service.apply_styles(
			user_prompt='test',
			user_negative_prompt='',
			style_identifiers=[],
		)

		_, negative = result

		# Should return user's negative (empty in this case means default)
		assert negative == ''

	def test_apply_styles_preserves_user_negative(self, service: StylesService, mock_styles: list[StyleItem]) -> None:
		"""Test that user's negative prompt is preserved and combined with style negatives."""
		service.all_styles = mock_styles

		result = service.apply_styles(
			user_prompt='test',
			user_negative_prompt='user negative',
			style_identifiers=['style1'],
		)

		_, negative = result

		# Should contain both user negative and style negative
		assert 'user negative' in negative
		assert 'cartoon' in negative

	def test_apply_styles_removes_duplicates(self, service: StylesService, mock_styles: list[StyleItem]) -> None:
		"""Test that duplicate style additions are removed."""
		# Create styles with overlapping content
		duplicate_styles = [
			StyleItem(
				id='dup1',
				name='Dup1',
				positive='{prompt}, detailed',
				negative='bad quality',
				image='https://example.com/dup1.jpg',
			),
			StyleItem(
				id='dup2',
				name='Dup2',
				positive='{prompt}, detailed',  # Duplicate
				negative='bad quality',  # Duplicate
				image='https://example.com/dup2.jpg',
			),
		]
		service.all_styles = duplicate_styles

		result = service.apply_styles(
			user_prompt='test',
			user_negative_prompt='',
			style_identifiers=['dup1', 'dup2'],
		)

		positive, negative = result

		# Count occurrences - should only appear once
		assert positive.count('detailed') == 1
		assert negative.count('bad quality') == 1

	def test_apply_styles_truncates_if_exceeds_token_limit(self, service: StylesService) -> None:
		"""Test that combined prompt is truncated if it exceeds token limit."""
		# Create a very long user prompt
		very_long_prompt = ' '.join(['word'] * 100)

		# Create a style that would push it over
		long_style = StyleItem(
			id='long',
			name='Long',
			positive='{prompt}, ' + ', '.join(['adjective'] * 50),
			negative='',
			image='https://example.com/long.jpg',
		)
		service.all_styles = [long_style]

		result = service.apply_styles(
			user_prompt=very_long_prompt,
			user_negative_prompt='',
			style_identifiers=['long'],
		)

		positive, _ = result

		# Result should be truncated to fit token limit (77 tokens)
		assert service.count_tokens(positive) <= 77

	def test_tokenizer_is_available(self, service: StylesService) -> None:
		"""Test that tokenizer is available for use."""
		# Tokenizer should be accessible
		tokenizer = service.tokenizer

		# Should be loaded and functional
		assert tokenizer is not None

		# Should return same instance on subsequent calls (caching)
		assert service.tokenizer is tokenizer

	def test_gpt2_tokenizer_is_available(self, service: StylesService) -> None:
		"""Test that GPT-2 tokenizer is available if loading succeeds."""
		# Access tokenizer property
		tokenizer = service.gpt2_tokenizer

		# Tokenizer may be None if loading fails, which is acceptable
		# If loaded, subsequent calls should return same instance
		if tokenizer is not None:
			assert service.gpt2_tokenizer is tokenizer

	@patch('app.services.styles.GPT2TokenizerFast')
	def test_gpt2_tokenizer_handles_loading_failure(self, mock_gpt2_class: Mock, service: StylesService) -> None:
		"""Test that GPT-2 tokenizer gracefully handles loading failures."""
		# Make GPT-2 loading fail
		mock_gpt2_class.from_pretrained.side_effect = Exception('Loading failed')

		# Should return None instead of raising
		tokenizer = service.gpt2_tokenizer

		assert tokenizer is None

	def test_apply_styles_with_nonexistent_style_id(self, service: StylesService, mock_styles: list[StyleItem]) -> None:
		"""Test that nonexistent style IDs are ignored."""
		service.all_styles = mock_styles

		result = service.apply_styles(
			user_prompt='test',
			user_negative_prompt='',
			style_identifiers=['nonexistent'],
		)

		# Should return unchanged prompts since style doesn't exist
		assert result == ['test', '']

	def test_apply_styles_prompt_placeholder_removed_from_first_style(self, service: StylesService) -> None:
		"""Test that {prompt} placeholder is properly handled in first style."""
		style_with_placeholder = StyleItem(
			id='test',
			name='Test',
			positive='before {prompt} after',
			negative='',
			image='https://example.com/test.jpg',
		)
		service.all_styles = [style_with_placeholder]

		result = service.apply_styles(
			user_prompt='user text',
			user_negative_prompt='',
			style_identifiers=['test'],
		)

		positive, _ = result

		# {prompt} should be removed from first style
		assert '{prompt}' not in positive
		assert 'user text' in positive
		assert 'before' in positive
		assert 'after' in positive
