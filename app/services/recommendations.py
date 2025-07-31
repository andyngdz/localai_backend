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
	RecommendationSectionType,
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
		default_recommend_section = self.get_recommended_section_type(device_capabilities)
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

	def get_recommended_section_type(self, capabilities: DeviceCapabilities) -> RecommendationSectionType:
		"""Get the recommended section type based on hardware capabilities"""

		if capabilities.max_gpu_gb >= 8:
			return RecommendationSectionType.HIGH_PERFORMANCE
		elif capabilities.max_gpu_gb >= 4:
			return RecommendationSectionType.STANDARD
		else:
			return RecommendationSectionType.LIGHTWEIGHT

	def is_section_recommended(self, section_id: RecommendationSectionType, capabilities: DeviceCapabilities) -> bool:
		"""Determine if a section should be recommended based on hardware"""

		recommended_section = self.get_recommended_section_type(capabilities)
		return section_id == recommended_section

	def get_default_selected_model(self, sections: List[ModelRecommendationSection]) -> str:
		"""Get the default selected model from recommendations"""

		# Find first recommended model in sections (recommended sections first)
		sorted_sections = sorted(sections, key=lambda s: (not s.is_recommended, s.id))

		for section in sorted_sections:
			# Look for recommended models first
			for model in section.models:
				if model.is_recommended:
					return model.id

			# If no recommended models, take first model from recommended section
			if section.is_recommended and section.models:
				return section.models[0].id

		# Fallback: first model from any section
		for section in sorted_sections:
			if section.models:
				return section.models[0].id

		# Ultimate fallback
		return DEFAULT_FALLBACK_MODEL


model_recommendation_service = ModelRecommendationService()
