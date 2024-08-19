from database.models.historico import Historico
from database.db import db

class DbManager:
    #guarda en la base de datos el estado de las peticiones junto con un mensaje
    def save_request(estado, doc1, doc2, mensaje, similitud):
        new_historico = Historico(estado, doc1, doc2, mensaje, similitud)
        db.session.add(new_historico)
        db.session.commit()

    def get_historico():
        peticiones = Historico.query.all()
        data = [
            {
                'Fecha': historico.fecha.strftime('%Y-%m-%d %H:%M:%S'),  # Asegurando que la fecha sea serializable a JSON
                'doc1': historico.doc1,
                'doc2': historico.doc2,
                'Estado': historico.estado,
                'Mensaje': historico.mensaje,
                'Similitud': historico.porcentaje_similitud
            }
            for historico in peticiones
        ]
        return data

        
