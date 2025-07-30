"""Model Recommendation Service"""

import logging
from typing import List

from sqlalchemy.orm import Session

from app.constants.recommendations import DEFAULT_FALLBACK_MODEL, SECTION_CONFIGS
from app.model_loader.max_memory import MaxMemoryConfig
from app.schemas.recommendations import (
	DeviceCapabilities,
	ModelRecommendationResponse,
	ModelRecommendationSection,
)
from app.services import device_service

logger = logging.getLogger(__name__)


class ModelRecommendationService:
	"""Service for generating hardware-based model recommendations"""

	def get_recommendations(self, db: Session) -> ModelRecommendationResponse:
		"""Get model recommendations based on current hardware configuration"""

		logger.info('Generating model recommendations based on hardware capabilities')

		memory_config = MaxMemoryConfig(db)
		device_capabilities = self.get_device_capabilities(memory_config)

		sections = self.build_recommendation_sections(device_capabilities)
		default_recommend_section = self.get_default_recommend_section(device_capabilities)
		default_selected_model = self.get_default_selected_model(sections)

		return ModelRecommendationResponse(
			sections=sections,
			default_recommend_section=default_recommend_section,
			default_selected_model=default_selected_model,
		)

	def get_device_capabilities(self, memory_config: MaxMemoryConfig) -> DeviceCapabilities:
		"""Extract device capabilities for recommendation logic"""

		return DeviceCapabilities(
			max_ram_gb=memory_config.max_ram,
			max_gpu_gb=memory_config.max_gpu,
			is_cuda=device_service.is_cuda,
			is_mps=device_service.is_mps,
			device_index=memory_config.device_index,
		)

	def build_recommendation_sections(self, capabilities: DeviceCapabilities) -> List[ModelRecommendationSection]:
		"""Build recommendation sections based on hardware capabilities"""

		sections = []

		# Build sections based on hardware capabilities
		for section_id, config in SECTION_CONFIGS.items():
			# Check if this section is applicable for current hardware
			if capabilities.max_gpu_gb >= config.min_gpu_gb:
				# Determine if this section should be recommended
				is_recommended = self.is_section_recommended(section_id, capabilities)

				# Create section with models from constants
				section = ModelRecommendationSection(
					id=section_id,
					name=config.name,
					description=config.description,
					is_recommended=is_recommended,
					models=config.models,
				)
				sections.append(section)

		logger.info(
			f'Generated {len(sections)} recommendation sections for '
			f'device capabilities: max_gpu_gb={capabilities.max_gpu_gb}'
		)

		return sections

	def is_section_recommended(self, section_id: str, capabilities: DeviceCapabilities) -> bool:
		"""Determine if a section should be recommended based on hardware"""

		if section_id == 'high-performance':
			return capabilities.max_gpu_gb >= 8
		elif section_id == 'standard':
			return 4 <= capabilities.max_gpu_gb < 8
		elif section_id == 'lightweight':
			return capabilities.max_gpu_gb < 4
		return False

	def get_default_recommend_section(self, capabilities: DeviceCapabilities) -> str:
		"""Determine the default recommended section based on hardware"""

		if capabilities.max_gpu_gb >= 8:
			return 'high-performance'
		elif capabilities.max_gpu_gb >= 4:
			return 'standard'
		else:
			return 'lightweight'

	def get_default_selected_model(self, sections: List[ModelRecommendationSection]) -> str:
		"""Get the default selected model from recommendations"""

		# Find first recommended model in recommended section
		for section in sections:
			if section.is_recommended:
				for model in section.models:
					if model.is_recommended:
						return model.id

		# Fallback to first model in first section
		if sections and sections[0].models:
			return sections[0].models[0].id

		# Ultimate fallback
		return DEFAULT_FALLBACK_MODEL


model_recommendation_service = ModelRecommendationService()
