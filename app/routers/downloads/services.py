import fnmatch
import json
from typing import List

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from config import CACHE_FOLDER


class DownloadService:
	api = HfApi()

	def download_model(self, id: str):
		components = self.parse_model_index(id)
		allow_patterns = ['model_index.json'] + [f'{c}/*' for c in components]
		files = self.list_files(id)
		ignore_patterns = ['*.bin'] if self.repo_has_safetensors_in_scopes(files, allow_patterns) else None

		local_dir = snapshot_download(
			repo_id=id,
			allow_patterns=allow_patterns,
			ignore_patterns=ignore_patterns,
			cache_dir=CACHE_FOLDER,
			max_workers=4,
		)

		return local_dir

	def repo_has_safetensors_in_scopes(self, files: List[str], scopes: List[str]) -> bool:
		for f in files:
			if f.endswith('.safetensors') and any(fnmatch.fnmatch(f, p) for p in scopes):
				return True
		return False

	def list_files(self, id: str) -> List[str]:
		info = self.api.repo_info(id)

		return [s.rfilename for s in info.siblings]

	def parse_model_index(self, id: str):
		model_index = hf_hub_download(
			repo_id=id,
			filename='model_index.json',
			repo_type='model',
		)

		with open(model_index, 'r', encoding='utf-8') as f:
			data = json.load(f)

		components = []

		for key, val in data.items():
			if isinstance(val, list) and val[0] is not None:
				components.append(key)

		return components


download_service = DownloadService()
