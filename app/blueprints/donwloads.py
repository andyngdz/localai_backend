"""Downloads Blueprint"""

import os
import threading
from flask import Blueprint, jsonify, logging, request
from pydantic import ValidationError

from app.schemas.core import ErrorResponse
from app.schemas.downloads import (
    DownloadRequest,
    DownloadStatus,
    DownloadStatusResponse,
    DownloadStatusStates,
)
from app.core.shared_data import download_statuses


logger = logging.create_logger(__name__)

MODEL_STORAGE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "models"
)

os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)

downloads = Blueprint("downloads", __name__)


def _download_model_in_background(model_id: str, hf_token: str = None):
    """Download model in a separate thread"""
    pass


@downloads.route("/", methods=["POST"])
def initiate_download():
    """Start download model"""
    try:
        request_data = DownloadRequest.model_validate_json(request.data)
    except ValidationError as e:
        logger.error("Validation error for download request: %s", e.errors())
        return (
            jsonify(
                ErrorResponse(
                    detail=e.errors(),
                    type="ValidationError",
                ).model_dump()
            ),
            400,
        )
    except TypeError as e:
        logger.error("Type error for download request: %s", str(e))
        return (
            jsonify(
                ErrorResponse(
                    detail=str(e),
                    type="TypeError",
                ).model_dump()
            ),
            400,
        )
    except ValueError as e:
        logger.error("Value error for download request: %s", str(e))
        return (
            jsonify(
                ErrorResponse(
                    detail=str(e),
                    type="ValueError",
                ).model_dump()
            ),
            400,
        )

    model_id = request_data.model_id
    hf_token = request_data.hf_token

    logger.info("API Request: Initiating download for model_id: %s", model_id)

    download_statuses[model_id] = DownloadStatus(model_id=model_id)

    thread = threading.Thread(
        target=_download_model_in_background, args=(model_id, hf_token)
    )
    thread.daemon = True
    thread.start()

    return (
        jsonify(
            DownloadStatusResponse(
                model_id=model_id,
                status=DownloadStatusStates.PENDING,
                message="Download started",
            )
        ),
        202,
    )
