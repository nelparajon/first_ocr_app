
class Config:
    #configuraciones de la base de datos SQLAlchemy
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    FLASK_TESTING = False
    DEBUG = True
    TESTING = False

class TestConfig(Config):
    #configuriaciones de la base de datos de prueba
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    TESTING = True
    DEBUG = True
