import os
import sys
import unittest 
from unittest.mock import patch
from PIL import Image
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from process_pdf.process_to_images import ProcessToImages

class ProcessDocTest(unittest.TestCase):
    def setUp(self):
        # Crear una instancia de la clase ProcessDocTest
        self.processor = ProcessToImages()


    #CONVERT TO GRAY TESTS
    def test_convert_rgb_to_gray(self):
        #Crear una imagen RGB de 3 canales
        img_rgb = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img_gray = self.processor.convert_to_gray(img_rgb)
        # Verificar que el resultado sea una imagen de 2 dimensiones (grayscale)
        self.assertEqual(len(img_gray.shape), 2)
    
    def test_convert_rgba_to_gray(self):
        #Crear una imagen RGBA de 4 canales
        img_rgba = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
        img_gray = self.processor.convert_to_gray(img_rgba)
        # Verificar que el resultado sea una imagen de 2 dimensiones (grayscale)
        self.assertEqual(len(img_gray.shape), 2)

    def test_convert_gray_image(self):
        img_gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        # Verificar que la función no cambie la imagen
        result = self.processor.convert_to_gray(img_gray)
        np.testing.assert_array_equal(result, img_gray)

    def test_empty_image(self):
        img_empty = np.array([])
        with self.assertRaises(ValueError):
            self.processor.convert_to_gray(img_empty)

    def test_invalid_type(self):
        invalid_input = "No es una imagen"
        with self.assertRaises(TypeError):
            self.processor.convert_to_gray(invalid_input)


    #THRESHOLD TESTS
    def test_valid_gray_image(self):
        """Test con una imagen válida en escala de grises"""
        img_gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        img_otsu = self.processor.thresh_otsu(img_gray)
        self.assertEqual(img_otsu.shape, img_gray.shape)
        #VerificaMOS que la imagen esté binarizada (contenga solo valores 0 o 255)
        self.assertTrue(np.array_equal(np.unique(img_otsu), [0, 255]))

    def test_none_image(self):
        """Test cuando la imagen es None"""
        # Verificar que se lanza una excepción de tipo ValueError
        with self.assertRaises(ValueError):
            self.processor.thresh_otsu(None)

    def test_invalid_type(self):
        """Test cuando el tipo de la imagen no es np.ndarray"""
        invalid_input = "Esto no es una imagen"
        # Verificar que se lanza una excepción de tipo TypeError
        with self.assertRaises(TypeError):
            self.processor.thresh_otsu(invalid_input)

    def test_empty_image(self):
        """Test cuando la imagen está vacía"""
        img_empty = np.array([])  # Imagen vacía
        # Verificar que se lanza una excepción de tipo ValueError
        with self.assertRaises(ValueError):
            self.processor.thresh_otsu(img_empty)

            
    #INVERT IMAGE TESTS
    def test_invert_valid_image(self):
        """Test para invertir una imagen válida en NumPy"""
        img_binary = np.random.choice([0, 255], size=(100, 100)).astype(np.uint8)
        inverted_image = self.processor.invert_image(img_binary)
        expected_image = cv2.bitwise_not(img_binary)
        np.testing.assert_array_equal(inverted_image, expected_image)

    def test_none_image(self):
        """Test para verificar que se lance ValueError si la imagen es None"""
        with self.assertRaises(ValueError) as context:
            self.processor.invert_image(None)
        self.assertEqual(str(context.exception), "No existe ninguna imagen para invertir su binario")

    def test_invalid_type(self):
        """Test para verificar que se lance TypeError con un tipo de dato no válido"""
        invalid_input = "Esto no es una imagen"
        with self.assertRaises(TypeError) as context:
            self.processor.invert_image(invalid_input)
        self.assertEqual(str(context.exception), "El tipo de dato de la imagen no es válido. Se esperaba un numpy.ndarray.")

    def test_empty_image(self):
        """Test para verificar que se lance ValueError si la imagen está vacía"""
        img_empty = np.array([], dtype=np.uint8)
        with self.assertRaises(ValueError) as context:
            self.processor.invert_image(img_empty)
        self.assertEqual(str(context.exception), "La imagen está vacía.")

    def test_invert_pil_image(self):
        """Test para verificar que una imagen PIL se convierte a NumPy pero lanza TypeError"""
        img_pil = Image.fromarray(np.random.choice([0, 255], size=(100, 100)).astype(np.uint8))
        with self.assertRaises(TypeError) as context:
            self.processor.invert_image(img_pil)
        self.assertEqual(str(context.exception), "El tipo de dato de la imagen no es válido. Se esperaba un numpy.ndarray.")


    #SEARCH CONTOURS TESTS
    def test_search_contours_valid_image(self):
        """Test para buscar contornos en una imagen binarizada válida"""
        img_binary = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(img_binary, (30, 30), (70, 70), 255, -1)
        contours = self.processor.search_contours(img_binary)
        self.assertGreater(len(contours), 0)
        #Verificamo que el primer contorno tiene 4 puntos
        self.assertEqual(len(contours[0]), 4)

    def test_none_image(self):
        """Test para verificar que se lance ValueError si la imagen es None"""
        with self.assertRaises(ValueError) as context:
            self.processor.search_contours(None)
        self.assertEqual(str(context.exception), "La imagen proporcionada no existe.")

    def test_invalid_type(self):
        """Test para verificar que se lance TypeError con un tipo de dato no válido"""
        invalid_input = "Esto no es una imagen"
        with self.assertRaises(TypeError) as context:
            self.processor.search_contours(invalid_input)
        self.assertEqual(str(context.exception), "El tipo de dato de la imagen no es válido. Se esperaba un numpy.ndarray.")

    def test_image_not_grayscale(self):
        """Test para verificar que se lance ValueError si la imagen no está en escala de grises"""
        img_rgb = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        with self.assertRaises(ValueError) as context:
            self.processor.search_contours(img_rgb)
        self.assertEqual(str(context.exception), "La imagen proporcionada no está en escala de grises.")

    def test_empty_image(self):
        """Test para verificar que una imagen vacía devuelva una lista vacía"""
        img_empty = np.zeros((100, 100), dtype=np.uint8)  #Imagen vacía (todo negro)
        contours = self.processor.search_contours(img_empty)
        self.assertEqual(len(contours), 0)


    #BLUR TESTS
    def test_blur_valid_grayscale_image(self):
        """Test para aplicar desenfoque en una imagen válida en escala de grises"""
        img_gray = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
        blurred_image = self.processor.blur_image(img_gray)
        
        # Verificar que la imagen desenfocada tiene las mismas dimensiones que la original
        self.assertEqual(img_gray.shape, blurred_image.shape)

    def test_blur_valid_color_image(self):
        """Test para aplicar desenfoque en una imagen válida en color"""
        img_color = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
        blurred_image = self.processor.blur_image(img_color)
        
        # Verificar que la imagen desenfocada tiene las mismas dimensiones que la original
        self.assertEqual(img_color.shape, blurred_image.shape)

    def test_invalid_type(self):
        """Test para verificar que se lance TypeError si la entrada no es un numpy.ndarray"""
        invalid_input = "Esto no es una imagen"
        with self.assertRaises(TypeError) as context:
            self.processor.blur_image(invalid_input)
        self.assertEqual(str(context.exception), "El tipo de dato de la imagen no es válido. Se esperaba un numpy.ndarray.")

    def test_invalid_dimensions_image(self):
        """Test para verificar que se lance ValueError si la imagen tiene dimensiones incorrectas"""
        img_invalid = np.random.randint(0, 256, size=(100,), dtype=np.uint8)  
        with self.assertRaises(ValueError) as context:
            self.processor.blur_image(img_invalid)
        self.assertEqual(str(context.exception), "La imagen proporcionada tiene un número de dimensiones incorrecto.")

    def test_empty_image(self):
        """Test para verificar que se maneje correctamente una imagen vacía"""
        img_empty = np.array([], dtype=np.uint8)
        with self.assertRaises(ValueError) as context:
            self.processor.blur_image(img_empty)
        self.assertEqual(str(context.exception), "La imagen proporcionada tiene un número de dimensiones incorrecto.")


    #KERNEL TESTS
    def test_kernel_image_valid(self):
        """Test para verificar que se genera un kernel de 5x5 correctamente"""
        img_valid = np.random.randint(0, 256, (100, 100), dtype=np.uint8)  # Creamos la imagen valida
        kernel = self.processor.kernel_image(img_valid)
        # Verificar que el kernel tiene las dimensiones correctas (5x5 en este caso)
        self.assertEqual(kernel.shape, (5, 5))
        self.assertTrue(np.array_equal(kernel, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))))

    def test_kernel_image_invalid_type(self):
        """Test para verificar que se lance TypeError si el tipo de la imagen es incorrecto"""
        invalid_input = "Esto no es una imagen"
        with self.assertRaises(TypeError) as context:
            self.processor.kernel_image(invalid_input)
        self.assertEqual(str(context.exception), "La imagen proporcionada debe ser un numpy.ndarray.")

    #DILATE IMAGE TESTS
    def test_dilate_valid_image_and_kernel(self):
        """Test para aplicar dilatación a una imagen válida con un kernel válido"""
        img_valid = np.random.randint(0, 256, (100, 100), dtype=np.uint8)  # Crear una imagen binarizada
        kernel_valid = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Crear un kernel válido
        
        dilated_image = self.processor.dilate_image(img_valid, kernel_valid)
        
        # Verificar que la imagen dilatada tiene las mismas dimensiones que la original
        self.assertEqual(img_valid.shape, dilated_image.shape)

    def test_dilate_kernel_none(self):
        """Test para verificar que se lance ValueError si el kernel es None"""
        img_valid = np.random.randint(0, 256, (100, 100), dtype=np.uint8)  #Creamos una imagen binarizada
        with self.assertRaises(ValueError) as context:
            self.processor.dilate_image(img_valid, None)
        self.assertEqual(str(context.exception), "La imagen o el kernel proporcionados son None.")

    def test_dilate_invalid_kernel_type(self):
        """Test para verificar que se lance TypeError si el kernel no es un numpy.ndarray"""
        img_valid = np.random.randint(0, 256, (100, 100), dtype=np.uint8)  #Crear una imagen binarizada
        invalid_kernel = "Esto no es un kernel"
        with self.assertRaises(TypeError) as context:
            self.processor.dilate_image(img_valid, invalid_kernel)
        self.assertEqual(str(context.exception), "Tanto la imagen como el kernel deben ser numpy.ndarray.")

    
    #FILTER PARAGRAPHS TEST 
    def test_filter_valid_contours(self):
        """Test para filtrar contornos válidos de una imagen"""
        img_valid = np.random.randint(0, 256, (500, 500), dtype=np.uint8)  # Crear una imagen de 500x500
        contours = [np.array([[10, 10], [10, 100], [100, 100], [100, 10]]),  # Contorno pequeño
                    np.array([[200, 200], [200, 400], [400, 400], [400, 200]]),  # Contorno grande
                    np.array([[300, 50], [300, 300], [500, 300], [500, 50]])]  # Contorno en la zona no válida
        
        valid_contours = self.processor.filter_paragraph_contours(img_valid, contours)
        
        # Verificar que solo se haya encontrado un contorno válido
        self.assertEqual(len(valid_contours), 1)
        self.assertEqual(valid_contours[0], (200, 200, 201, 201))  # El contorno grande

    def test_image_none(self):
        """Test para verificar que se lance TypeError si la imagen es None"""
        contours = []
        with self.assertRaises(TypeError) as context:
            self.processor.filter_paragraph_contours(None, contours)
        self.assertEqual(str(context.exception), "La imagen debe ser un numpy.ndarray o PIL.Image.")

    def test_invalid_image_type(self):
        """Test para verificar que se lance TypeError si la imagen no es un numpy.ndarray o PIL.Image"""
        contours = []
        invalid_image = "Esto no es una imagen"
        with self.assertRaises(TypeError) as context:
            self.processor.filter_paragraph_contours(invalid_image, contours)
        self.assertEqual(str(context.exception), "La imagen debe ser un numpy.ndarray o PIL.Image.")

    def test_invalid_image_dimensions(self):
        """Test para verificar que se lance ValueError si la imagen tiene dimensiones incorrectas"""
        invalid_image = np.random.randint(0, 256, (100,), dtype=np.uint8)  # Crear una imagen con una dimensión
        contours = []
        with self.assertRaises(ValueError) as context:
            self.processor.filter_paragraph_contours(invalid_image, contours)
        self.assertEqual(str(context.exception), "La imagen proporcionada tiene dimensiones inválidas.")


    #EXTRACT IMAGE WITH CONTOURS TESTS
    @patch.object(ProcessToImages, 'convert_to_gray')
    @patch.object(ProcessToImages, 'blur_image')
    @patch.object(ProcessToImages, 'thresh_otsu')
    @patch.object(ProcessToImages, 'invert_image')
    @patch.object(ProcessToImages, 'kernel_image')
    @patch.object(ProcessToImages, 'dilate_image')
    @patch.object(ProcessToImages, 'search_contours')
    @patch.object(ProcessToImages, 'filter_paragraph_contours')
    def test_extract_image_with_valid_contours(self, mock_filter_paragraph_contours, mock_search_contours, mock_dilate_image, 
                                               mock_kernel_image, mock_invert_image, mock_thresh_otsu, mock_blur_image, 
                                               mock_convert_to_gray):
        """Test para extraer contornos válidos de imágenes y dibujarlos"""
        images = {
            1: np.random.randint(0, 256, (100, 100), dtype=np.uint8),
            2: np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        }

        # Configurar los mocks para devolver valores simulados
        mock_convert_to_gray.return_value = images[1]
        mock_blur_image.return_value = images[1]
        mock_thresh_otsu.return_value = images[1]
        mock_invert_image.return_value = images[1]
        mock_kernel_image.return_value = np.ones((5, 5), np.uint8)
        mock_dilate_image.return_value = images[1]
        mock_search_contours.return_value = [np.array([[10, 10], [10, 50], [50, 50], [50, 10]])]
        mock_filter_paragraph_contours.return_value = [(10, 10, 40, 40)]  # Contorno válido

        # Llamar al método extract_image_with_contours
        images_with_contours, contours_positions = self.processor.extract_image_with_contours(images)

        # Verificar que el método devuelva resultados válidos
        self.assertIn(1, images_with_contours)
        self.assertIn(1, contours_positions)
        self.assertEqual(contours_positions[1], [{'x': 10, 'y': 10, 'width': 40, 'height': 40}])

    def test_extract_invalid_image_type(self):
        """Test para verificar que se lance TypeError si alguna imagen no es un numpy.ndarray"""
        images = {
            1: np.random.randint(0, 256, (100, 100), dtype=np.uint8),
            2: "Esto no es una imagen válida"
        }
        with self.assertRaises(TypeError):
            self.processor.extract_image_with_contours(images)

    def test_extract_empty_images(self):
        """Test para verificar que el diccionario de imágenes no está vacío"""
        images = {}
        with self.assertRaises(ValueError):
            self.processor.extract_image_with_contours(images)


    #FILTER COUNTOURS TITLES TESTS
    def test_filter_valid_titles(self):
        """Test para verificar que los contornos de títulos válidos se filtran correctamente"""
        image = np.ones((1000, 800), dtype=np.uint8) * 255  # Imagen blanca

        # Dibujar contornos simulados
        # Simular un título (contorno válido)
        cv2.rectangle(image, (50, 50), (700, 90), 0, -1)  # Contorno negro
        
        # Simular un contorno más pequeño que no es un título
        cv2.rectangle(image, (100, 200), (150, 220), 0, -1)  # Contorno negro

        # Convertir la imagen a objeto PIL para pasarlo a la función
        pil_image = Image.fromarray(image)
        
        # Detectar los contornos en la imagen
        contours, _ = cv2.findContours(cv2.Canny(image, 100, 200), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_titles = self.processor.filter_title_contours(
            pil_image, 
            contours, 
            min_width_ratio=0.12, 
            max_height_ratio=0.07, 
            density_threshold=0.25
        )
        
        # Solo el primer contorno debería ser considerado válido como título
        self.assertEqual(len(valid_titles), 1)
        self.assertEqual(len(valid_titles), 1)

        # Permitir un margen de error de ±2 píxeles para cada coordenada/dimensión dandole tolerancia de error a openCV
        x, y, w, h = valid_titles[0]
        expected_x, expected_y, expected_w, expected_h = (50, 50, 650, 40)

        self.assertTrue(abs(x - expected_x) <= 2, f"x: {x} no está dentro de la tolerancia")
        self.assertTrue(abs(y - expected_y) <= 2, f"y: {y} no está dentro de la tolerancia")
        self.assertTrue(abs(w - expected_w) <= 2, f"w: {w} no está dentro de la tolerancia")
        self.assertTrue(abs(h - expected_h) <= 2, f"h: {h} no está dentro de la tolerancia")

    def test_filter_no_valid_contours(self):
        """Test para verificar que no se detecten contornos si no cumplen los criterios de ancho, altura o densidad"""
        img_valid = np.random.randint(0, 256, (500, 500), dtype=np.uint8)  # Crear una imagen válida de 500x500
        contours = [np.array([[50, 50], [50, 200], [100, 200], [100, 50]])]  # Contorno que no cumple con los criterios

        valid_titles = self.processor.filter_title_contours(img_valid, contours, min_width_ratio=0.12, max_height_ratio=0.07, density_threshold=0.25)

        # No se debe detectar ningún contorno válido
        self.assertEqual(len(valid_titles), 0)


    #EXTRAER TITULOS TESTS
    @patch.object(ProcessToImages, 'convert_to_gray')
    @patch.object(ProcessToImages, 'blur_image')
    @patch.object(ProcessToImages, 'thresh_otsu')
    @patch.object(ProcessToImages, 'invert_image')
    @patch.object(ProcessToImages, 'kernel_image')
    @patch.object(ProcessToImages, 'dilate_image')
    @patch.object(ProcessToImages, 'search_contours')
    @patch.object(ProcessToImages, 'filter_title_contours')
    def test_extraer_titulos_valid_images(self, mock_filter_title_contours, mock_search_contours, mock_dilate_image, 
                                          mock_kernel_image, mock_invert_image, mock_thresh_otsu, mock_blur_image, 
                                          mock_convert_to_gray):
        """Test para verificar que los títulos se extraen correctamente de imágenes válidas"""
        # Crear un diccionario de imágenes válidas
        images = {
            1: np.random.randint(0, 256, (100, 100), dtype=np.uint8),
            2: np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        }

        # Configurar los mocks para devolver valores simulados
        mock_filter_title_contours.return_value = [(10, 10, 40, 40)]  # Títulos válidos
        mock_search_contours.return_value = [np.array([[10, 10], [50, 10], [50, 50], [10, 50]])]
        mock_dilate_image.return_value = images[1]
        mock_kernel_image.return_value = np.ones((5, 5), np.uint8)
        mock_invert_image.return_value = images[1]
        mock_thresh_otsu.return_value = images[1]
        mock_blur_image.return_value = images[1]
        mock_convert_to_gray.return_value = images[1]

        # Llamar al método extraer_titulos
        image_with_titles, titles_position = self.processor.extraer_titulos(images)

        # Verificar que se dibujaron los contornos de títulos en las imágenes
        self.assertIn(1, image_with_titles)
        self.assertIn(1, titles_position)
        self.assertEqual(titles_position[1], [{'x': 10, 'y': 10, 'width': 40, 'height': 40}])

    def test_extraer_titulos_empty_images(self):
        """Test para verificar que se lance ValueError si el diccionario de imágenes está vacío"""
        images = {}
        with self.assertRaises(ValueError) as context:
            self.processor.extraer_titulos(images)
        self.assertEqual(str(context.exception), "El diccionario de imágenes está vacío.")

    def test_extraer_titulos_invalid_image_type(self):
        """Test para verificar que se lance TypeError si alguna imagen no es un numpy.ndarray o PIL.Image"""
        images = {
            1: np.random.randint(0, 256, (100, 100), dtype=np.uint8),
            2: "Esto no es una imagen válida"
        }
        with self.assertRaises(TypeError) as context:
            self.processor.extraer_titulos(images)
        self.assertEqual(str(context.exception), "La imagen en la página 2 no es un numpy.ndarray ni una PIL.Image.")


    #PROCESS DOCUMENTS TESTS
    @patch.object(ProcessToImages, 'extraer_titulos')
    @patch.object(ProcessToImages, 'extract_image_with_contours')
    def test_process_valid_images(self, mock_extract_image_with_contours, mock_extraer_titulos):
        """Test para procesar un diccionario de imágenes válidas"""
        images = {
            1: np.random.randint(0, 256, (100, 100), dtype=np.uint8),
            2: np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        }

        # Simular la respuesta de las funciones extraer_titulos y extract_image_with_contours
        mock_extraer_titulos.return_value = (images, {1: 'title_pos_1', 2: 'title_pos_2'})
        mock_extract_image_with_contours.return_value = (images, {1: 'para_pos_1', 2: 'para_pos_2'})

        # Llamar al método process_document
        images_with_paragraphs, titles_position, paragraphs_position = self.processor.process_document(images)

        # Verificar que las funciones hayan sido llamadas
        mock_extraer_titulos.assert_called_once_with(images)
        mock_extract_image_with_contours.assert_called_once_with(images)

        # Verificar que el resultado es el esperado
        self.assertEqual(images_with_paragraphs, images)
        self.assertEqual(titles_position, {1: 'title_pos_1', 2: 'title_pos_2'})
        self.assertEqual(paragraphs_position, {1: 'para_pos_1', 2: 'para_pos_2'})

    def test_process_invalid_image_type(self):
        """Test para verificar que se lance TypeError si alguna imagen no es un numpy.ndarray"""
        images = {
            1: np.random.randint(0, 256, (100, 100), dtype=np.uint8),
            2: "Esto no es una imagen válida"
        }
        with self.assertRaises(TypeError):
            self.processor.process_document(images)

    def test_process_empty_images(self):
        """Test para verificar que se lance ValueError si el diccionario de imágenes está vacío"""
        images = {}
        with self.assertRaises(ValueError):
            self.processor.process_document(images)


    

if __name__ == "__main__":
    unittest.main()