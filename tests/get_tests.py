from datetime import datetime
import os
import sys
import unittest 
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database.db import db
from database.models.historico import Historico

from cliente.app import create_app
from cliente.routes import service_get_historico

class GetRequestsTest(unittest.TestCase):
    
    def setUp(self):
        self.app = create_app('testing')
        self.app_client = self.app.test_client()
        self.app.testing = True

        
        with self.app.app_context():
            db.create_all()

            #registro para pruebas
            sample_data = Historico(
                doc1="documento1.pdf",
                doc2="documento2.pdf",
                estado="Solicitud exitosa. Código: 200",
                mensaje="Subida y procesamiento de archivos completada.",
                porcentaje_similitud=95.75
            )
            
            db.session.add(sample_data)
            db.session.commit()

            # Guardar la fecha generada automáticamente para usar en las aserciones
            self.expected_fecha = sample_data.fecha.strftime('%Y-%m-%d %H:%M:%S')

    #eliminamos las tablas 
    def tearDown(self):
        with self.app.app_context():
            db.session.remove()
            db.drop_all()

    #Test de la solicitud GET
    def test_get_historico(self):
        response = self.app_client.get('/historico')
        self.assertEqual(response.status_code, 200)
        
        expected_data = {'Peticiones': [{
            "Estado": "Solicitud exitosa. Código: 200",
            "Fecha": self.expected_fecha,
            "Mensaje": "Subida y procesamiento de archivos completada.", 
            "Similitud": '95.75',
            "doc1": "documento1.pdf",
            "doc2": "documento2.pdf",     
        }]}

        
        self.assertIn('Peticiones', response.json)
        self.assertEqual(response.json, expected_data) #comparamos los datos esperados con lo que devuelve el json de la peticion
        print("Test de petición GET HISTORICO Completada")
                

if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit as e:
        print(f"SystemExit: {e}")
        raise
    


        

        