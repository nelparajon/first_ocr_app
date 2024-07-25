from flask import Flask
from routes import analize

def create_app():
    app = Flask(__name__)  # Añadir DATABASE=ruta/a/la/bd cuando se cree
    app.register_blueprint(analize)
    return app


