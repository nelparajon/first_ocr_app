import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from cliente.routes import analize
from database.db import db
from database import config
from cliente.error_handler import ErrorHandler
from flask_cors import CORS
from flask_migrate import Migrate



def create_app(config_name='default'):
    app = Flask(__name__)
    
    # Configuración de la aplicación basada en el entorno
    if config_name == 'testing':
        app.config.from_object(config.TestConfig)
    else:
        database_uri = os.getenv('SQLALCHEMY_DATABASE_URI')
        if not database_uri:
            raise RuntimeError("La variable de entorno 'SQLALCHEMY_DATABASE_URI' no está definida")
        
        app.config['SQLALCHEMY_DATABASE_URI'] = database_uri
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = config.Config.SQLALCHEMY_TRACK_MODIFICATIONS
        app.config['DEBUG'] = config.Config.DEBUG
        app.config['TESTING'] = config.Config.FLASK_TESTING
    
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

    db.init_app(app)

    with app.app_context():
        from database.models.historico import Historico  # Importación del modelo
        from database.db_manager import DbManager
        db.create_all()
        app.register_blueprint(analize)
        ErrorHandler.register(app)

    return app


