"""Latent upscaling for hires fix."""

import torch

from app.schemas.hires_fix import InterpolationMode, UpscalerType
from app.services import logger_service

logger = logger_service.get_logger(__name__, category='Upscaler')

MIN_SCALE_FACTOR = 1.0


class LatentUpscaler:
	"""Handles upscaling of latent tensors for hires fix.

	Upscales latent tensors in latent space using interpolation methods.
	This is more memory-efficient than decoding to PIL and re-encoding.
	"""

	def upscale(
		self,
		latents: torch.Tensor,
		scale_factor: float,
		upscaler_type: UpscalerType = UpscalerType.LATENT,
	) -> torch.Tensor:
		"""Upscale latents using interpolation.

		Args:
			latents: Input latent tensor [batch, channels, height, width]
			scale_factor: Upscaling factor (e.g., 2.0 for 2x)
			upscaler_type: Upscaling method (Latent, Latent (nearest), Latent (nearest-exact))

		Returns:
			Upscaled latent tensor

		Raises:
			ValueError: If scale_factor <= 1.0
		"""
		if scale_factor <= MIN_SCALE_FACTOR:
			raise ValueError(f'scale_factor must be > {MIN_SCALE_FACTOR}, got {scale_factor}')

		original_height = latents.size(2)
		original_width = latents.size(3)
		logger.info(
			f'Upscaling latents from {original_height}x{original_width} by {scale_factor}x using {upscaler_type.value}'
		)

		upscaled = self._interpolate(latents, scale_factor, upscaler_type)

		new_height = upscaled.size(2)
		new_width = upscaled.size(3)
		logger.info(f'Upscaled to {new_height}x{new_width}')

		return upscaled

	def _interpolate(
		self,
		latents: torch.Tensor,
		scale_factor: float,
		upscaler_type: UpscalerType,
	) -> torch.Tensor:
		"""Apply torch interpolation with correct parameters.

		Args:
			latents: Input latent tensor
			scale_factor: Upscaling factor
			upscaler_type: Upscaler type enum

		Returns:
			Interpolated tensor
		"""
		mode, align_corners = self._get_interpolation_params(upscaler_type)

		return torch.nn.functional.interpolate(
			latents,
			scale_factor=scale_factor,
			mode=mode,
			align_corners=align_corners,
		)

	def _get_interpolation_params(self, upscaler_type: UpscalerType) -> tuple[InterpolationMode, bool | None]:
		"""Get interpolation parameters for upscaler type.

		Args:
			upscaler_type: Upscaler type enum

		Returns:
			Tuple of (mode, align_corners)
		"""
		mapping = {
			UpscalerType.LATENT: (InterpolationMode.BILINEAR, False),
			UpscalerType.LATENT_NEAREST: (InterpolationMode.NEAREST, None),
			UpscalerType.LATENT_NEAREST_EXACT: (InterpolationMode.NEAREST_EXACT, None),
		}
		return mapping[upscaler_type]


latent_upscaler = LatentUpscaler()
