"""Main entry point for the LocalAI Backend application."""

import logging

from flask import Flask, jsonify, send_from_directory

from app.blueprints.downloads import downloads
from app.blueprints.hardware import hardware
from app.blueprints.models import models
from app.blueprints.users import users
from app.services.socket_io import socketio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

app = Flask(__name__)
app.register_blueprint(users, url_prefix='/users')
app.register_blueprint(models, url_prefix='/models')
app.register_blueprint(downloads, url_prefix='/downloads')
app.register_blueprint(hardware, url_prefix='/hardware')
socketio.init_app(app, cors_allowed_origins='*')


@app.route('/favicon.ico')
def favicon():
    """Serve the favicon.ico file from the static folder."""
    # send_from_directory serves a file from a specified directory.
    # app.static_folder automatically points to the 'static' folder Flask found.
    # 'favicon.ico' is the filename within that folder.
    static_folder = app.static_folder or 'static'

    return send_from_directory(
        static_folder, 'favicon.ico', mimetype='image/vnd.microsoft.icon'
    )


@app.route('/')
def health_check():
    """Health check endpoint to verify if the server is running."""
    return jsonify({'status': 'healthy', 'message': 'LocalAI Backend is running!'})


if __name__ == '__main__':
    socketio.run(app, debug=True)
