from database.models.historico import Historico
from database.db import db

class DbManager:
    #guarda en la base de datos el estado de las peticiones junto con un mensaje
    def save_request(estado, mensaje, similitud):
        new_historico = Historico(estado, mensaje, similitud)
        db.session.add(new_historico)
        db.session.commit()

    def get_historico():
        peticiones = Historico.query.all()
        data = [
            {
                'Fecha': historico.fecha.strftime('%Y-%m-%d %H:%M:%S'),  # Asegurando que la fecha sea serializable a JSON
                'Estado': historico.estado,
                'Mensaje': historico.mensaje,
                'Similitud': historico.porcentaje_similitud
            }
            for historico in peticiones
        ]
        return data
        
