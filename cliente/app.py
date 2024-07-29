from flask import Flask
from cliente.routes import analize
from database.db import db
from database import config
from cliente.error_handler import ErrorHandler

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = config.SQLALCHEMY_DATABASE_URI
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = config.SQLALCHEMY_TRACK_MODIFICATIONS
    app.config['DEBUG'] = config.DEBUG
    app.config ['TESTING'] = config.FLASK_TESTING

    db.init_app(app)

    with app.app_context():
        from database.models.historico import Historico #importacion del modelo
        from database.db_manager import DbManager
        db.create_all()
        app.register_blueprint(analize)
        ErrorHandler.register(app)

    return app


