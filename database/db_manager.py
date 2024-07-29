from database.models.historico import Historico
from database.db import db

class DbManager:
    #guarda en la base de datos el estado de las peticiones junto con un mensaje
    def save_request(estado, mensaje):
        new_historico = Historico(estado, mensaje)
        db.session.add(new_historico)
        db.session.commit()
        
