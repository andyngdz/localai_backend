from asyncio import Task
from collections import defaultdict
from typing import Dict

# Global dict to hold progress
# Example structure:
# {
#   "stable-diffusion-v1-5": {
#       "unet/model.bin": {"downloaded": 1024, "total": 2048},
#       ...
#   },
#   ...
# }
download_progresses: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(dict)


download_tasks: Dict[str, Task] = {}
