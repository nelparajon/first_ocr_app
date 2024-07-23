import unittest
import sys
import os

# Agregar el directorio principal del proyecto al PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ocr_pdf')))
from ocr_pdf.vectorizer import Vectorizer

class VectorizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.vectorizer = Vectorizer()
        return super().setUp()
    
    def test_vectorizing(self):
        lems = [['run', 'good', 'geese', 'foot', 'cat', 'big', 'leaf'], ['talk', 'bad', 'child', 'mouse', 'run']]
        matrix = self.vectorizer.vectorize_doc(lems)
        
        expected_matrix = [
            [0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0],
            [1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1]
        ]
        self.assertTrue((matrix.toarray() == expected_matrix).all(), "El test de vectorización no ha tenido éxito" )
        
        print("Vectorización realizada con éxito")
    
    def test_vectorize_doc_type_error(self):
        with self.assertRaises(TypeError) as cm:
            self.tokenizer.vectorize_doc("Esto no es una lista de tokens")
        self.assertEqual(str(cm.exception), "Se esperaba una lista de cadenas, pero se recibió str")

        with self.assertRaises(TypeError) as cm:
            self.tokenizer.vectorize_doc([1, 2, 3])
        self.assertEqual(str(cm.exception), "Se esperaba una lista de cadenas, pero se recibió list")

    def test_vectorize_doc_value_error(self):
        with self.assertRaises(ValueError) as cm:
            self.tokenizer.vectorize_doc(["!!!"])
        self.assertEqual(str(cm.exception), "Se esperaba una lista de cadenas de tokens, se recibió list")

    
    def test_similarity(self):
        pass

if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit as e:
        print(f"SystemExit: {e}")
        raise