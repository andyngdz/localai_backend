"""Shared data for the app."""

from typing import Dict

from app.schemas.downloads import DownloadStatus

# --- This dictionary will keep track of download statuses ---
# Key: model_id (the full Hugging Face ID like "runwayml/stable-diffusion-v1-5")
# Value: a dictionary with keys like "status", "progress", "error", "path", "current_file"
# In a production app, this would typically be stored in a persistent way
# (e.g., a database, Redis, or a dedicated task queue like Celery)
# because this in-memory dictionary will reset if the Flask server restarts.
download_statuses: Dict[str, DownloadStatus] = {}
