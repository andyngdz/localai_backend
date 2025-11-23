import os
from pathlib import Path
from typing import Literal, NotRequired, Optional, TypedDict, Union, cast

from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor

from app.constants.model_loader import ModelLoadingStrategy
from app.cores.model_manager.pipeline_manager import DiffusersPipeline
from app.schemas.model_loader import ModelLoadFailed
from app.services import device_service, logger_service
from app.socket import socket_service
from config import CACHE_FOLDER

from .cancellation import CancellationToken
from .progress import emit_progress

logger = logger_service.get_logger(__name__, category='ModelLoad')


class SingleFileStrategy(TypedDict):
	type: Literal[ModelLoadingStrategy.SINGLE_FILE]
	checkpoint_path: str


class PretrainedStrategy(TypedDict):
	type: Literal[ModelLoadingStrategy.PRETRAINED]
	use_safetensors: bool
	variant: NotRequired[str]


Strategy = Union[SingleFileStrategy, PretrainedStrategy]


def find_single_file_checkpoint(model_path: str) -> Optional[str]:
	if not os.path.exists(model_path):
		return None

	checkpoint_files = list(Path(model_path).glob('*.safetensors'))
	if checkpoint_files:
		checkpoint_path = str(checkpoint_files[0])
		logger.info(f'Found single-file checkpoint: {checkpoint_path}')
		return checkpoint_path

	return None


def find_checkpoint_in_cache(model_cache_path: str) -> Optional[str]:
	if not os.path.exists(model_cache_path):
		return None

	snapshots_dir = os.path.join(model_cache_path, 'snapshots')
	if not os.path.exists(snapshots_dir):
		return None

	snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
	if not snapshots:
		return None

	latest_snapshot = os.path.join(snapshots_dir, snapshots[0])
	return find_single_file_checkpoint(latest_snapshot)


def build_loading_strategies(checkpoint_path: Optional[str]) -> list[Strategy]:
	strategies: list[Strategy] = []
	if checkpoint_path:
		strategies.append(
			{
				'type': ModelLoadingStrategy.SINGLE_FILE,
				'checkpoint_path': checkpoint_path,
			}
		)

	strategies.append({'type': ModelLoadingStrategy.PRETRAINED, 'use_safetensors': True})
	strategies.append({'type': ModelLoadingStrategy.PRETRAINED, 'use_safetensors': False})
	strategies.append({'type': ModelLoadingStrategy.PRETRAINED, 'use_safetensors': True, 'variant': 'fp16'})
	strategies.append({'type': ModelLoadingStrategy.PRETRAINED, 'use_safetensors': False, 'variant': 'fp16'})

	return strategies


def _load_single_file(
	checkpoint: str,
	safety_checker: StableDiffusionSafetyChecker,
	feature_extractor: CLIPImageProcessor,
) -> DiffusersPipeline:
	from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
	from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline

	errors = []

	for pipeline_class in [StableDiffusionXLPipeline, StableDiffusionPipeline]:
		try:
			logger.debug(f'Trying {pipeline_class.__name__} for single-file checkpoint')

			from_single_file = getattr(pipeline_class, 'from_single_file')

			pipe = cast(
				DiffusersPipeline,
				from_single_file(
					checkpoint,
					torch_dtype=device_service.torch_dtype,
				),
			)

			if pipeline_class.__name__ == 'StableDiffusionXLPipeline' and getattr(pipe, 'tokenizer_2', None) is None:
				raise ValueError('StableDiffusionXLPipeline loaded without tokenizer_2 (likely SD 1.5 checkpoint)')

			if hasattr(pipe, 'safety_checker'):
				pipe.safety_checker = safety_checker

			if hasattr(pipe, 'feature_extractor'):
				pipe.feature_extractor = feature_extractor

			logger.info(f'Successfully loaded with {pipeline_class.__name__}')
			return pipe
		except Exception as error:
			errors.append(f'{pipeline_class.__name__}: {error}')

	raise ValueError(f'Failed to load single-file checkpoint {checkpoint}. Tried: {", ".join(errors)}')


def _load_pretrained(
	id: str,
	params: PretrainedStrategy,
	safety_checker: StableDiffusionSafetyChecker,
	feature_extractor: CLIPImageProcessor,
) -> DiffusersPipeline:
	load_params = {key: value for key, value in params.items() if key != 'type'}

	return AutoPipelineForText2Image.from_pretrained(
		id,
		cache_dir=CACHE_FOLDER,
		low_cpu_mem_usage=True,
		torch_dtype=device_service.torch_dtype,
		safety_checker=safety_checker,
		feature_extractor=feature_extractor,
		**load_params,
	)


def _get_strategy_type(strategy: Strategy) -> ModelLoadingStrategy:


	return cast(ModelLoadingStrategy, strategy['type'])








def _load_strategy_pipeline(


	id: str,


	strategy: Strategy,


	strategy_type: ModelLoadingStrategy,


	safety_checker: StableDiffusionSafetyChecker,


	feature_extractor: CLIPImageProcessor,


) -> DiffusersPipeline:


	if strategy_type == ModelLoadingStrategy.SINGLE_FILE:


		checkpoint_path = cast(SingleFileStrategy, strategy)['checkpoint_path']


		if not checkpoint_path:


			raise ValueError('Missing checkpoint path for single-file strategy')


		return _load_single_file(checkpoint_path, safety_checker, feature_extractor)





	return _load_pretrained(


		id,


		cast(PretrainedStrategy, strategy),


		safety_checker,


		feature_extractor,


	)


def execute_loading_strategies(
	id: str,
	strategies: list[Strategy],
	safety_checker: StableDiffusionSafetyChecker,
	feature_extractor: CLIPImageProcessor,
	cancel_token: Optional[CancellationToken],
) -> DiffusersPipeline:
	last_error: Optional[Exception] = None

	for idx, strategy in enumerate(strategies, 1):
		if cancel_token:
			cancel_token.check_cancelled()

		emit_progress(id, 5, 'Loading model weights...')

		try:
			strategy_type = _get_strategy_type(strategy)
			logger.info(f'Trying loading strategy {idx}/{len(strategies)} ({strategy_type}): {strategy}')

			pipe = _load_strategy_pipeline(
				id,
				strategy,
				strategy_type,
				safety_checker,
				feature_extractor,
			)

			logger.info(f'Successfully loaded model using strategy {idx}')
			return pipe
		except Exception as error:  # pragma: no cover - log/continue until last failure
			last_error = error
			logger.warning(f'Strategy {idx} failed: {error}')

			continue

	error_msg = f'Failed to load model {id} with all strategies. Last error: {last_error}'
	logger.error(error_msg)
	socket_service.model_load_failed(ModelLoadFailed(id=id, error=str(last_error)))

	if last_error is not None:
		raise last_error
	raise RuntimeError(error_msg)


__all__ = [
	'SingleFileStrategy',
	'PretrainedStrategy',
	'Strategy',
	'find_single_file_checkpoint',
	'find_checkpoint_in_cache',
	'build_loading_strategies',
	'execute_loading_strategies',
]
