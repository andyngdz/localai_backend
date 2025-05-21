from flask import Blueprint;

models = Blueprint('models', __name__)

@models.route('/list', methods=['GET'])
def list_models():
    return "List of models"
