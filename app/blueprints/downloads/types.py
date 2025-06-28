from typing import Awaitable, Callable

ProgressCallbackType = Callable[[str, int, int], Awaitable[None]]
