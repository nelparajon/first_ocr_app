from database.models.historico import Historico
from database.db import db

class DbManager:
    
    def save_request(estado, mensaje):
        new_historico = Historico(estado, mensaje)
        db.session.add(new_historico)
        db.session.commit()
        
