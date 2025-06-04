"""Blueprint for handling drivers in the application."""

import logging

from flask import Blueprint


logger = logging.getLogger(__name__)

drivers = Blueprint("drivers", __name__)
