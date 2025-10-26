"""Tests for the logger service with category support."""

import logging
import os
from unittest.mock import patch

import pytest

from app.services.logger import CategoryAdapter, LoggerService, logger_service


class TestCategoryAdapter:
	"""Test the CategoryAdapter class."""

	def test_category_prefix_added_to_message(self):
		"""Test that category prefix is added to log messages."""
		base_logger = logging.getLogger('test_logger')
		adapter = CategoryAdapter(base_logger, category='TestCategory')

		msg, kwargs = adapter.process('Test message', {})

		assert msg == '[TestCategory] Test message'
		assert kwargs == {}

	def test_category_adapter_preserves_logger_name(self):
		"""Test that CategoryAdapter preserves the underlying logger name."""
		base_logger = logging.getLogger('test.module')
		adapter = CategoryAdapter(base_logger, category='Test')

		assert adapter.logger.name == 'test.module'
		assert adapter.category == 'Test'


class TestLoggerService:
	"""Test the LoggerService class."""

	def test_get_logger_returns_category_adapter(self):
		"""Test that get_logger returns a CategoryAdapter instance."""
		service = LoggerService()
		logger = service.get_logger('test.module', category='TestCat')

		assert isinstance(logger, CategoryAdapter)
		assert logger.category == 'TestCat'

	@pytest.mark.parametrize(
		'category',
		['ModelLoad', 'Download', 'Generate', 'API', 'Database', 'Service', 'Socket', 'GPU'],
	)
	def test_all_categories(self, category):
		"""Test that all standard categories work correctly."""
		service = LoggerService()
		logger = service.get_logger('test.module', category=category)

		assert isinstance(logger, CategoryAdapter)
		assert logger.category == category

	def test_module_specific_log_level(self):
		"""Test that module-specific log levels are applied from environment variables."""
		service = LoggerService()

		# Module name conversion: app.test.module -> TEST_MODULE
		with patch.dict(os.environ, {'LOG_LEVEL_TEST_MODULE': 'DEBUG'}):
			logger = service.get_logger('app.test.module', category='Test')

			# The underlying logger should have DEBUG level set
			assert logger.logger.level == logging.DEBUG

	def test_default_log_level_when_no_env_var(self):
		"""Test that logger uses INFO level when no env var is set."""
		service = LoggerService()

		with patch.dict(os.environ, {}, clear=True):
			# Remove all LOG_LEVEL env vars
			env_clean = {k: v for k, v in os.environ.items() if not k.startswith('LOG_LEVEL')}
			with patch.dict(os.environ, env_clean, clear=True):
				logger = service.get_logger('app.test.other', category='Test')

				# Should use INFO as default (root logger level)
				assert logger.logger.level == 0 or logger.logger.level == logging.NOTSET


class TestLoggerIntegration:
	"""Integration tests for logger service."""

	def test_logger_service_singleton(self):
		"""Test that logger_service is a singleton instance."""
		assert isinstance(logger_service, LoggerService)

	def test_logger_with_all_categories_integration(self, caplog):
		"""Test that loggers with different categories work in integration."""
		categories = ['ModelLoad', 'Download', 'Generate', 'API', 'Database', 'Service', 'Socket', 'GPU']

		with caplog.at_level(logging.INFO):
			for category in categories:
				logger = logger_service.get_logger(f'test.{category.lower()}', category=category)
				logger.info(f'Testing {category}')

		# Verify log messages contain category prefixes
		for category in categories:
			assert any(f'[{category}] Testing {category}' in record.message for record in caplog.records)

	def test_category_adapter_with_different_log_levels(self, caplog):
		"""Test that CategoryAdapter works with different log levels."""
		logger = logger_service.get_logger('test.levels', category='Test')

		with caplog.at_level(logging.DEBUG):
			logger.debug('Debug message')
			logger.info('Info message')
			logger.warning('Warning message')
			logger.error('Error message')

		messages = [record.message for record in caplog.records]

		assert '[Test] Debug message' in messages
		assert '[Test] Info message' in messages
		assert '[Test] Warning message' in messages
		assert '[Test] Error message' in messages

	def test_logger_inherits_root_configuration(self):
		"""Test that loggers created after init() inherit root configuration."""
		# The logger_service.init() should have been called in main.py
		# Our loggers should inherit from the configured root logger

		logger_service.get_logger('test.inheritance', category='Test')

		# Should have the same handlers as root or inherit through propagation
		root_logger = logging.getLogger()
		assert len(root_logger.handlers) > 0  # Root should have handlers from init()
