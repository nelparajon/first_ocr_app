import unittest
import sys
import os

# Agregar el directorio principal del proyecto al PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ocr_pdf.tokenizer import Tokenizer

class TokenizerTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()
        return super().setUp()


    def test_tokenizer(self):
        text = "Vectorization is a crucial process in the field of Natural Language Processing (NLP)"
        expected_tokens = ['vectorization', 'crucial', 'process', 'field', 'natural', 'language', 'processing', 'nlp']
        result = self.tokenizer.tokenize_texts(text)
        self.assertEqual(result, expected_tokens, "La función no tokenizó el texto como se esperaba")
        print("Test de tokenización completado con éxito")

if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit as e:
        print(f"SystemExit: {e}")
        raise