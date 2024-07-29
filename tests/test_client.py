import base64
import os
import sys
import unittest
from cliente.app import create_app
from cliente.encoder import Encoder

#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ocr_pdf')))

class TestClient(unittest.TestCase):
    def setUp(self):
          self.app = create_app() #app flask
          self.app_context = self.app.app_context() #contexto Bd para pruebas con ella
          self.app_client = self.app.test_client() #cliente de pruebas de  flask
          self.app_context.push()
    
    def tearDown(self):
         self.app_context.pop()

    #llamada al servicio - main page
    def test_call_main_page(self):
        response  = self.app_client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'ANALIZADOR DE TEXTOS', response.data)
    
    #comprobación de la funcionalidad principal
    def test_route_analizar_textos(self):
        

        file_1_content = Encoder.encode_file_b64(r"C:\Users\Nel\Documents\texto_1.pdf" )
        file_2_content = Encoder.encode_file_b64(r"C:\Users\Nel\Documents\texto_2.pdf" )

        data = {
            'file_1': file_1_content,
            'file_2': file_2_content
        }
        response = self.app_client.post('/analizar_documentos', json=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Subida y procesamiento de archivos completada.', response.data)

    
    """def test_error_wrong_method(self):
        response = self.app_client.get('/analizar_documentos')
        self.assertEqual(response.status_code, 405)
        self.assertIn(b"Error: método no soportado", response.data)"""

    if __name__ == '__main_-':
        try:
            unittest.main()
        except SystemExit as e:
            print(f"SystemExit: {e}")
            raise

         

