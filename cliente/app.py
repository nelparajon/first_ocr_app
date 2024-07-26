from flask import Flask
from routes import analize
from database.db import db
import database.config as config

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = config.SQLALCHEMY_DATABASE_URI
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = config.SQLALCHEMY_TRACK_MODIFICATIONS

    db.init_app(app)

    with app.app_context():
        from database.models.historico import Historico #importacion del modelo
        db.create_all()
        app.register_blueprint(analize)

    return app


