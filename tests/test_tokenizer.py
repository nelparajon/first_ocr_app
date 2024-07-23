import unittest
import sys
import os

# Agregar el directorio principal del proyecto al PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ocr_pdf')))

from ocr_pdf.tokenizer import Tokenizer
from setup_nltk import setup_nltk

class TokenizerTest(unittest.TestCase):
    def setUp(self):
        setup_nltk()
        self.tokenizer = Tokenizer()
        super().setUp()

    # Verificar si hace lo buscado
    def test_tokenizer(self):
        text = "Vectorization is a crucial process in the field of Natural Language Processing"
        expected_tokens = ['vectorization', 'crucial', 'process', 'field', 'natural', 'language', 'processing']
        result = self.tokenizer.tokenize_texts(text)
        self.assertEqual(result, expected_tokens, "La función no tokenizó el texto como se esperaba")
        print("Test de tokenización completado con éxito")
  
    def test_type_error(self):
        with self.assertRaises(TypeError) as cm:
            self.tokenizer.tokenize_texts(["Esto no es una cadena de texto"])
        self.assertEqual(str(cm.exception), "Se esperaba una cadena, pero se recibió list")

        with self.assertRaises(TypeError) as cm:
            self.tokenizer.tokenize_texts(12345)
        self.assertEqual(str(cm.exception), "Se esperaba una cadena, pero se recibió int")

    def test_value_error(self):
        with self.assertRaises(ValueError) as cm:
            self.tokenizer.tokenize_texts("")
        self.assertEqual(str(cm.exception), "El texto no está limpio de signos de puntuación, caracteres especiales y mayúsculas. Tokenización fallida.")
        
        with self.assertRaises(ValueError) as cm:
            self.tokenizer.tokenize_texts("!!!")
        self.assertEqual(str(cm.exception), "El texto no está limpio de signos de puntuación, caracteres especiales y mayúsculas. Tokenización fallida.")



        
if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit as e:
        print(f"SystemExit: {e}")
        raise