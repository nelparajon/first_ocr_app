import base64
import io
import os
import sys
import unittest
from unittest.mock import patch

#Agregamos la ruta al sys.path para importar correctamente la app y las rutas
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cliente.app import create_app
from cliente.routes import analyze_service_route
from database.db import db
from database.models.historico import Historico

class TestAnalizarDocumentos(unittest.TestCase):

    def setUp(self):
        
        self.app = create_app('testing')
        self.app_client = self.app.test_client()
        self.app.testing = True
        #Ruta al archivo PDF de prueba
        self.pdf_path = os.path.join(os.path.dirname(__file__), '../docs_pruebas/lematizacion.pdf')

        #creamos las tablas en la BD
        with self.app.app_context():
            db.create_all()

    def tearDown(self):
        #eliminamos las tablas de la BD
        with self.app.app_context():
            db.session.remove()
            db.drop_all()

    #test de la solicitud completa
    @patch('cliente.routes.Encoder.encode_file_b64')
    def test_upload_and_analyze_files(self, mock_encode_file_b64):
        with open(self.pdf_path, "rb") as file:
            file_content = file.read()
        
        mock_encode_file_b64.return_value = base64.b64encode(file_content).decode('utf-8')
        
        #datos para la solititud
        data = {
            'pdf1': (io.BytesIO(file_content), 'file1.pdf'),
            'pdf2': (io.BytesIO(file_content), 'file2.pdf')
        }
        response = self.app_client.post(analyze_service_route, content_type='multipart/form-data', data=data)

        #verificacion del json de la respuesta
        self.assertEqual(response.status_code, 200)
        response_json = response.get_json()
        self.assertIn('mensaje', response_json)
        self.assertEqual(response_json['mensaje'], 'Subida y procesamiento de archivos completada.')
        self.assertIn('Código', response_json)
        self.assertEqual(response_json['Código'], 200)
        self.assertIn('Similitud', response_json)
        self.assertAlmostEqual(float(response_json['Similitud']), 100.00, places=2)
        
        #verificamos que los datos se han registrado en la base de datos
        with self.app.app_context():
            historico_entry = Historico.query.filter_by(doc1='file1.pdf', doc2='file2.pdf').first()
            self.assertIsNotNone(historico_entry, "No se encontró la entrada en la base de datos.")
            self.assertEqual(historico_entry.estado, "Solicitud exitosa. Código: 200")
            self.assertEqual(historico_entry.mensaje, 'Subida y procesamiento de archivos completada.')
            self.assertAlmostEqual(historico_entry.porcentaje_similitud, 100.00, places=2)

        print("Test Valid Files realizado con éxito y datos registrados en la base de datos verificados")
    
    #simulamos una solicitud sin archivos
    def test_missing_files(self):
        response = self.app_client.post(analyze_service_route)
        response_json, status_code = response.json, response.status_code
        
        self.assertEqual(status_code, 400)
        self.assertIn('No se encontraron los archivos', response_json['error'])
        print("Test Missing Files realizado con éxito")


    #test para verificar si la decodifición errónea de los archivos da un error
    @patch('cliente.routes.Encoder.encode_file_b64', side_effect=Exception("Decoding error"))
    def test_decoding_error(self, mock_encode_file_b64):
        
        data = {
            'pdf1': (io.BytesIO(b'somepdfcontent1'), 'file1.pdf'),
            'pdf2': (io.BytesIO(b'somepdfcontent2'), 'file2.pdf')
        }
        response = self.app_client.post(analyze_service_route, content_type='multipart/form-data', data=data)
        #Verificamos la respuesta de error
        self.assertEqual(response.status_code, 500)
        self.assertIn(b'Error al decodificar el archivo', response.data)
        print("Test Decoding Error realizado con éxito")

    #verificamos si da un error al validar el formato de los archivos
    @patch('cliente.routes.Encoder.validate_pdf')
    def test_invalid_pdf(self, mock_validate_pdf):
        mock_validate_pdf.return_value = False
        data = {
        'pdf1': (io.BytesIO(b'somecontent'), 'file1.pdf'),
        'pdf2': (io.BytesIO(b'somecontent'), 'file2.pdf')
    }
        #verificamos la respuesta de error
        response = self.app_client.post(analyze_service_route, content_type='multipart/form-data', data=data)
        self.assertEqual(response.status_code, 422)
        self.assertIn('Error al validar el formato de los archivos', response.get_json()['mensaje'])
        print("Test invalid PDF realizado con éxito")

    #verificamos que si el json con el contenido del pdf está vacío lanza un error
    def test_empty_files_b64(self):
        data = {
            'file1': {'content': '', 'name': 'file1.pdf'},
            'file2': {'content': '', 'name': 'file2.pdf'}
        }
        response = self.app_client.post('/analizar_documentos_b64', json=data)
        self.assertEqual(response.status_code, 400)
        expected_response = {
            "mensaje": "El contenido de los archivos no puede estar vacío",
            "Código": 400,
            "Similitud": "0.0"
        }
        self.assertEqual(response.json, expected_response)
        print("Test para archivos vacíos realizado con éxito")

    #verificamos que si el contenido no está en base64 nos da un error
    def test_invalid_base64(self):
        data = {
            'file1': {'content': 'contenido invalido', 'name': 'file1.pdf'},
            'file2': {'content': 'contenido invalido', 'name': 'file2.pdf'}
        }
        response = self.app_client.post('/analizar_documentos_b64', json=data)
        self.assertEqual(response.status_code, 400)
        self.assertIn('Error al decodificar el archivo', response.json['error'])
        print("Test para base64 inválido realizado con éxito")


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit as e:
        print(f"SystemExit: {e}")
        raise

