# Plan: Handle missing `model_index.json`

1. **Repository fallback logic**

   - Update `HuggingFaceRepository.get_components` to catch `EntryNotFoundError` (and similar) from `hf_hub_download`.
   - Return an empty list instead of raising so callers can decide how to handle missing `model_index.json` files.
   - Keep JSON parsing behavior unchanged so malformed files still raise and surface clearly.

2. **DownloadService scope fallback**

   - In `DownloadService.download_model`, detect when `get_components` returns an empty list.
   - When no components exist, log a warning and fall back to downloading all files in the repository (rather than just `model_index.json`).
   - Ensure `get_ignore_components` still works by defaulting scopes to `['*']` in fallback mode.

3. **File selection adjustments**

   - Simplify the file filtering logic to make it easy to compare the “components only” flow with the fallback “download all files” flow.
   - Guarantee deterministic ordering and size tracking regardless of which flow runs.

4. **Tests**
   - Add coverage for the fallback path, verifying that downloads still proceed when `model_index.json` is missing.
   - Confirm that `get_components` gracefully handles missing files without raising.
