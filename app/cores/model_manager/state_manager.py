"""Model loading state machine."""

import logging
from enum import Enum

logger = logging.getLogger(__name__)


class ModelState(Enum):
	"""Model loading states for tracking operations."""

	IDLE = 'idle'  # No model loaded
	LOADING = 'loading'  # Model load in progress
	LOADED = 'loaded'  # Model successfully loaded
	UNLOADING = 'unloading'  # Model unload in progress
	CANCELLING = 'cancelling'  # Load operation being cancelled
	ERROR = 'error'  # Previous operation failed


class StateTransitionReason(Enum):
	"""Predefined reasons for state transitions."""

	LOAD_REQUESTED = 'load requested'
	LOAD_COMPLETED = 'load completed successfully'
	LOAD_FAILED = 'load failed with error'
	LOAD_CANCELLED = 'load cancelled by user'

	UNLOAD_REQUESTED = 'unload requested'
	UNLOAD_COMPLETED = 'unload completed successfully'
	UNLOAD_FAILED = 'unload failed with error'

	CANCELLATION_REQUESTED = 'cancellation requested'
	CANCELLATION_COMPLETED = 'cancellation completed'

	ERROR_OCCURRED = 'error occurred'
	RESET_FROM_ERROR = 'reset from error state'
	RESET_FROM_CANCELLING = 'reset from cancelling state'

	INITIALIZED = 'initialized'
	ALREADY_LOADED = 'already loaded'


class StateManager:
	"""Manages model loading state transitions with validation."""

	def __init__(self) -> None:
		self.state: ModelState = ModelState.IDLE

	def get_state(self) -> ModelState:
		"""Get current model state (thread-safe read).

		Returns:
			Current ModelState
		"""
		return self.state

	def set_state(self, new_state: ModelState, reason: StateTransitionReason) -> None:
		"""Set model state with logging.

		Args:
			new_state: New state to transition to
			reason: Predefined reason for transition
		"""
		old_state = self.state
		self.state = new_state
		logger.info(f'Model state: {old_state.value} → {reason.value}')

	def can_transition_to(self, target_state: ModelState) -> bool:
		"""Check if transition to target state is valid from current state.

		Args:
			target_state: State to transition to

		Returns:
			True if transition is valid, False otherwise
		"""
		valid_transitions = {
			ModelState.IDLE: {ModelState.LOADING},
			ModelState.LOADING: {ModelState.LOADED, ModelState.IDLE, ModelState.ERROR, ModelState.CANCELLING},
			ModelState.LOADED: {ModelState.UNLOADING},
			ModelState.UNLOADING: {ModelState.IDLE, ModelState.ERROR},
			ModelState.CANCELLING: {ModelState.IDLE, ModelState.ERROR},
			ModelState.ERROR: {ModelState.IDLE, ModelState.LOADING},
		}
		return target_state in valid_transitions.get(self.state, set())


# Singleton instance
state_manager = StateManager()
