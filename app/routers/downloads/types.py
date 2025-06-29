from types import CoroutineType
from typing import Any, Callable

ProgressCallbackType = Callable[[str, int, int], CoroutineType[Any, Any, None]]
