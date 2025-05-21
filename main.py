from flask import Flask

from app.blueprints.models import models

app = Flask(__name__)
app.register_blueprint(models, url_prefix='/models')

@app.route('/')
def index():
    return "Welcome to the LocalAI Backend"

if __name__ == '__main__':
    app.run(debug=True)