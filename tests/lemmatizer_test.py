import unittest
import sys
import os

# Agregar el directorio principal del proyecto al PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ocr_pdf.lemmatizer import Lemmatizer

class LemmatizerTest(unittest.TestCase):
    def setUp(self):
        self.lemmatizer = Lemmatizer()
        return super().setUp()
    
    def test_lemmatization(self):
        tokens = ['running', 'better', 'geese', 'feet', 'cats', 'bigger', 'leaves']
        expected_lemmas = ['run', 'good', 'geese', 'foot', 'cat', 'big', 'leaf']
        result = self.lemmatizer.lemmatizing_words(tokens)
        self.assertEqual(result, expected_lemmas, "El text de lematizacion no se pudo completar con éxito")
        print("Lematización completada con éxito")

if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit as e:
        print(f"SystemExit: {e}")
        raise
