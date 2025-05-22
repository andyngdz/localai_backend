import logging
from flask import Flask, jsonify, send_from_directory

from app.blueprints.models import models

logging.getLogger().setLevel(logging.INFO)
app = Flask(__name__)
app.register_blueprint(models, url_prefix="/models")


@app.route("/favicon.ico")
def favicon():
    # send_from_directory serves a file from a specified directory.
    # app.static_folder automatically points to the 'static' folder Flask found.
    # 'favicon.ico' is the filename within that folder.
    return send_from_directory(
        app.static_folder, "favicon.ico", mimetype="image/vnd.microsoft.icon"
    )


@app.route("/")
def health_check():
    return jsonify({"status": "healthy", "message": "LocalAI Backend is running!"})


if __name__ == "__main__":
    app.run(debug=True)
