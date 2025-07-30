"""Model Recommendation Constants"""

from typing import Dict

from app.schemas.recommendations import ModelRecommendationItem, RecommendationSectionType, SectionConfig

# High Performance Models
HIGH_PERFORMANCE_MODELS = [
	ModelRecommendationItem(
		id='stabilityai/stable-diffusion-xl-base-1.0',
		name='Stable Diffusion XL',
		description='Latest high-quality image generation model',
		memory_requirement_gb=6,
		model_size='6.9GB',
		tags=['xl', 'high-quality', 'latest'],
		is_recommended=True,
	),
	ModelRecommendationItem(
		id='stabilityai/stable-diffusion-xl-refiner-1.0',
		name='SDXL Refiner',
		description='Refiner model for enhanced image quality',
		memory_requirement_gb=6,
		model_size='6.1GB',
		tags=['xl', 'refiner', 'enhancement'],
		is_recommended=False,
	),
]

# Standard Models
STANDARD_MODELS = [
	ModelRecommendationItem(
		id='runwayml/stable-diffusion-v1-5',
		name='Stable Diffusion 1.5',
		description='Reliable and widely-used image generation model',
		memory_requirement_gb=4,
		model_size='4.3GB',
		tags=['stable', 'reliable', 'popular'],
		is_recommended=True,
	),
	ModelRecommendationItem(
		id='stabilityai/stable-diffusion-2-1',
		name='Stable Diffusion 2.1',
		description='Improved version with better prompt adherence',
		memory_requirement_gb=5,
		model_size='5.1GB',
		tags=['improved', 'better-prompts'],
		is_recommended=False,
	),
]

# Lightweight Models
LIGHTWEIGHT_MODELS = [
	ModelRecommendationItem(
		id='segmind/small-sd',
		name='Small Stable Diffusion',
		description='Compact model for quick generation',
		memory_requirement_gb=2,
		model_size='1.8GB',
		tags=['compact', 'fast', 'efficient'],
		is_recommended=True,
	),
	ModelRecommendationItem(
		id='segmind/tiny-sd',
		name='Tiny Stable Diffusion',
		description='Ultra-compact model for minimal resource usage',
		memory_requirement_gb=1,
		model_size='983MB',
		tags=['tiny', 'minimal', 'ultra-fast'],
		is_recommended=False,
	),
]

# Section Configurations
SECTION_CONFIGS: Dict[str, SectionConfig] = {
	RecommendationSectionType.HIGH_PERFORMANCE: SectionConfig(
		name='High Performance',
		description='Advanced models for powerful hardware',
		models=HIGH_PERFORMANCE_MODELS,
		min_gpu_gb=8,
	),
	RecommendationSectionType.STANDARD: SectionConfig(
		name='Standard',
		description='Balanced performance and quality models',
		models=STANDARD_MODELS,
		min_gpu_gb=4,
	),
	RecommendationSectionType.LIGHTWEIGHT: SectionConfig(
		name='Lightweight',
		description='Optimized models for efficient generation',
		models=LIGHTWEIGHT_MODELS,
		min_gpu_gb=0,
	),
}

# Default fallback model
DEFAULT_FALLBACK_MODEL = 'runwayml/stable-diffusion-v1-5'
