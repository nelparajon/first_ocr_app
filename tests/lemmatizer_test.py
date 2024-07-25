import unittest
import sys
import os
from unittest.mock import patch

# Agregar el directorio principal del proyecto al PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from process_pdf.lemmatizer import Lemmatizer

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
    
    def test_get_wordnet_pos_type_error(self):
        with self.assertRaises(TypeError) as cm:
            self.lemmatizer.get_wordnet_pos(123)
        self.assertEqual(str(cm.exception), "Se esperaba una cadena de texto, pero se recibió int")

    def test_lemmatizing_words_type_error(self):
        with self.assertRaises(TypeError) as cm:
            self.lemmatizer.lemmatizing_words('not a list')
        self.assertEqual(str(cm.exception), "Los tokens deben de ser una lista de cadenas, se recibió str")

    def test_lemmatizion_words_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            self.lemmatizer.lemmatizing_words([1,2,3])
        self.assertEqual(str(ctx.exception), "Todos los elementos de la lista deben ser cadenas de texto.")
        

    

if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit as e:
        print(f"SystemExit: {e}")
        raise
