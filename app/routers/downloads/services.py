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
		ignore_patterns = self.bins_to_ignore(files, allow_patterns)

		local_dir = snapshot_download(
			repo_id=id,
			allow_patterns=allow_patterns,
			ignore_patterns=ignore_patterns,
			cache_dir=CACHE_FOLDER,
			max_workers=4,
		)

		return local_dir

	def bins_to_ignore(self, files: List[str], scopes: List[str]) -> List[str]:
		"""
		Return a list of .bin files that should be ignored if a corresponding
		.safetensors file with the same base name exists in the given scopes.

		Example:
			- If both "unet/model.bin" and "unet/model.safetensors" exist,
				then "unet/model.bin" will be ignored.
			- If only "vae/model.bin" exists (no safetensors), it will be kept.
		"""
		# Filter files that are inside the provided scopes (e.g. "unet/*", "vae/*")
		in_scope = [f for f in files if any(fnmatch.fnmatch(f, p) for p in scopes)]

		# Collect base names of all safetensors files in scope
		safetensors_bases = {f.removesuffix('.safetensors') for f in in_scope if f.endswith('.safetensors')}

		# Return .bin files that have a matching .safetensors base
		return [f for f in in_scope if f.endswith('.bin') and f.removesuffix('.bin') in safetensors_bases]

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
