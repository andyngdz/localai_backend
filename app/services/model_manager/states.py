from collections import defaultdict
from multiprocessing import Process
from typing import Dict

download_processes: Dict[str, Process] = defaultdict(Process)
