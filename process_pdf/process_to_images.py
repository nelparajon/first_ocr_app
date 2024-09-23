import os
import sys
import cv2
import numpy as np

from pdf2image import convert_from_bytes, convert_from_path
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from process_pdf.ocr_producer import OCRProducer
from cliente.encoder import Encoder

class ProcessToImages:

    def __init__(self, output_dir='./output_images'):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    #para pruebas
    def convert_pdf_to_image_from_file(self, file):
        """
        Convierte un archivo PDF a una serie de imágenes.

        Args:
            file (str): Ruta al archivo PDF que se convertirá.

        Returns:
            dict: Un diccionario donde las claves son el número de página y los valores son las imágenes correspondientes.

        Raises:
            FileNotFoundError: Si el archivo no se encuentra en la ruta proporcionada.
            OSError: Si ocurre un problema al abrir o procesar el archivo PDF.
            Exception: Si ocurre un error inesperado durante la conversión.
        """
        try:
            # Verificar si el archivo es una cadena (ruta)
            if not isinstance(file, str):
                raise TypeError("La ruta del archivo debe ser una cadena.")

            # Convertir el PDF a imágenes
            imgs = convert_from_path(file)
            
            # Crear un diccionario para almacenar las imágenes por número de página
            images = {}
            for i, img in enumerate(imgs):
                images[i + 1] = img
            
            print(images)
            return images

        except FileNotFoundError:
            print(f"El archivo {file} no se encuentra.")
            raise

        except OSError as e:
            print(f"Error al abrir o procesar el archivo PDF: {e}")
            raise

        except TypeError as te:
            print(f"Error de tipo de archivo: {te}")
            raise

        except Exception as e:
            print(f"Ha ocurrido un error inesperado: {e}")
            raise


    #pruebas para las imagenes de los parrafos
    def save_single_image(self, image, name):
        """
        Guarda una imagen en formato PNG en el directorio de salida especificado.

        Args:
            image (np.ndarray or Image.Image): La imagen a guardar en formato NumPy array o PIL Image.
            name (str): El nombre del archivo (sin extensión) con el que se guardará la imagen.

        Returns:
            None

        Raises:
            TypeError: Si la imagen no está en un formato válido (ni NumPy array ni PIL Image).
            ValueError: Si la ruta de salida no está configurada correctamente.
            Exception: Si ocurre un error inesperado durante el guardado de la imagen.
        """
        try:
            # Verificar si la imagen es un NumPy array, si no lo es, convertirla a NumPy
            if not isinstance(image, np.ndarray):
                if isinstance(image, Image.Image):  # Si es una imagen PIL, convertirla a NumPy array
                    image = np.array(image)
                else:
                    raise TypeError("La imagen debe ser un numpy.ndarray o PIL.Image.")

            # Asegurarse de que el directorio de salida está configurado
            if not hasattr(self, 'output_dir') or not os.path.isdir(self.output_dir):
                raise ValueError("El directorio de salida no está configurado correctamente o no existe.")
            
            # Guardar la imagen en formato PNG en el directorio de salida
            file_path = os.path.join(self.output_dir, f'{name}.png')
            cv2.imwrite(file_path, image)
            print(f"Imagen guardada exitosamente como {file_path}")

        except TypeError as te:
            print(f"Error de tipo: {str(te)}")
            raise

        except ValueError as ve:
            print(f"Error de valor: {str(ve)}")
            raise

        except Exception as e:
            print(f"Ha ocurrido un error inesperado al guardar la imagen: {str(e)}")
            raise


    #se convierte en un array numpy para poder escalar a grises
    def image_to_nparray(self, image):
        print(f"Tipo de imagen recibido: {type(image)}")
        if isinstance(image, np.ndarray):
            return image
        
        elif isinstance(image, Image.Image):
            return np.array(image)
        
        else:
            try:
                return np.array(image)
            except Exception as e:
                raise ValueError(f"No se puede convertir la imagen a ndarray. Tipo no soportado: {type(image)}")

    #convierte la imagen a escala de grises
    def convert_to_gray(self, image):
        """
        Convierte una imagen en formato NumPy array o PIL a escala de grises.

        Args:
            image (np.ndarray or Image.Image): La imagen a convertir a escala de grises.

        Returns:
            np.ndarray: La imagen convertida a escala de grises.

        Raises:
            ValueError: Si la imagen está vacía o es None.
            TypeError: Si la imagen no está en un formato soportado (ni NumPy array ni PIL).
            cv2.error: Si ocurre un error con OpenCV durante la conversión.
        """
        try:
            if not isinstance(image, (np.ndarray, Image.Image)):
                raise TypeError("La imagen no está en un formato soportado. Debe ser un NumPy array o una imagen de PIL.")
            # Convertir la imagen a NumPy array si es necesario
            img = self.image_to_nparray(image)
            
            # Verificar si la imagen es válida
            if img is None or img.size == 0:
                raise ValueError("La imagen está vacía o es None, no se puede convertir a escala de grises.")
            
            # Convertir imagen a escala de grises si es de tipo RGB o RGBA
            if len(img.shape) == 3:  # La imagen tiene canales de color
                if img.shape[2] == 4:  # Si tiene un canal alfa (RGBA), convertir a RGB
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Si ya está en escala de grises (2 dimensiones), mantenerla igual
            elif len(img.shape) == 2:  # Imagen ya en escala de grises
                img_gray = img
            
            # Si el formato no es compatible
            else:
                raise ValueError("Formato de imagen no soportado.")
            
            return img_gray

        except ValueError as ve:
            print(f"Error de valor: {ve}")
            raise

        except TypeError as te:
            print(f"Error de tipo: {te}")
            raise

        except cv2.error as e:
            print(f"Error de OpenCV durante la conversión a escala de grises: {e}")
            raise

        except Exception as e:
            print(f"Error inesperado: {e}")
            raise

    #threshold otsu automatico
    def thresh_otsu(self, image):
        """
        Aplica umbralización de Otsu a la imagen proporcionada.

        Args:
            image (np.ndarray): Imagen en escala de grises en formato NumPy array.

        Returns:
            np.ndarray: Imagen binarizada usando el umbral de Otsu.

        Raises:
            ValueError: Si la imagen es None o no es válida.
            TypeError: Si la imagen no es un numpy.ndarray.
            cv2.error: Si ocurre un error con OpenCV durante el proceso de umbralización.
            Exception: Para cualquier otro error inesperado.
        """
        try:
            # Verificar si la imagen es None
            if image is None:
                raise ValueError("La imagen proporcionada es None.")
            
            # Verificar si la imagen es un array de NumPy
            if not isinstance(image, np.ndarray):
                raise TypeError("El tipo de dato de la imagen no es válido. Se esperaba un numpy.ndarray.")
            
            # Verificar si la imagen está vacía
            if image.size == 0:
                raise ValueError("La imagen está vacía.")
            
            # Aplicar umbralización de Otsu
            _, img_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return img_otsu

        except cv2.error as e:
            print(f"Error de OpenCV durante la umbralización: {e}")
            raise
        except ValueError as ve:
            print(f"Valor incorrecto: {ve}")
            raise
        except TypeError as te:
            print(f"Tipo incorrecto: {te}")
            raise
        except Exception as ex:
            print(f"Ocurrió un error inesperado: {ex}")
            raise
   
    #invierte el binario de la imagen
    def invert_image(self, image):
        """
        Invierte los valores binarios de una imagen.

        Args:
            image (np.ndarray): Imagen binarizada en formato NumPy array.

        Returns:
            np.ndarray: Imagen con los valores binarios invertidos.

        Raises:
            ValueError: Si la imagen es None o está vacía.
            TypeError: Si la imagen no es un numpy.ndarray.
            cv2.error: Si ocurre un error con OpenCV durante el proceso de inversión.
            Exception: Para cualquier otro error inesperado.
        """
        
        try:
            if image is None:
                raise ValueError("No existe ninguna imagen para invertir su binario")
            if not isinstance(image, (np.ndarray,)):
                raise TypeError("El tipo de dato de la imagen no es válido. Se esperaba un numpy.ndarray.")
            if image.size == 0:
                raise ValueError("La imagen está vacía.")
            
            inverted_image = cv2.bitwise_not(image)
            return inverted_image

        except cv2.error as e:
            print(f"error de opencv durante el proceso de inversión del binario de la imagen: {e}")
            raise
        except TypeError as te:
            print(f"Tipo de dato incorrecto durante el proceso de inversión del binario de la imagen:{te}")
            raise
        except Exception as ex:
            print(f"Error inesperado durante el proceso de inversión del binario de la imagen: {ex}")
            raise
    
    #comprueba los contornos
    def search_contours(self, image):
        """
        Busca contornos en una imagen binarizada.

        Args:
            image (np.ndarray): Imagen binarizada en formato NumPy array.

        Returns:
            list: Lista de contornos encontrados en la imagen.

        Raises:
            ValueError: Si la imagen es None, no está en escala de grises o es inválida.
            TypeError: Si la imagen no es un numpy.ndarray.
            cv2.error: Si ocurre un error con OpenCV durante la búsqueda de contornos.
            Exception: Para cualquier otro error inesperado.
        """
        try:
            if image is None:
                raise ValueError("La imagen proporcionada no existe.")
            if not isinstance(image, (np.ndarray,)):
                raise TypeError("El tipo de dato de la imagen no es válido. Se esperaba un numpy.ndarray.")
            
            if len(image.shape) != 2:
                raise ValueError("La imagen proporcionada no está en escala de grises.")
            
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return contours
        except cv2.error as e:
            print(f"Error de OpenCV durante la búsqueda de contornos: {e}")
            raise
        except ValueError as ve:
            print(f"Valor incorrecto durante la busqueda de contornos: {ve}")
            raise
        except TypeError as te:
            print(f"Tipo incorrecto durante la busqueda de contornos: {te}")
            raise
        except Exception as ex:
            print(f"Ocurrió un error durante la busqueda de contornos: {ex}")
            raise

    #añade blur a la imagen para procesarla
    def blur_image(self, image):
        """
        Aplica un desenfoque gaussiano a la imagen proporcionada.

        Args:
            image (np.ndarray): Imagen en formato NumPy array.

        Returns:
            np.ndarray: Imagen desenfocada usando un filtro gaussiano.

        Raises:
            ValueError: Si la imagen es None o tiene un número incorrecto de dimensiones.
            TypeError: Si la imagen no es un numpy.ndarray.
            cv2.error: Si ocurre un error con OpenCV durante el desenfoque.
            Exception: Para cualquier otro error inesperado.
        """
        try:
            if image is None:
                raise ValueError("La imagen proporcionada no existe.")
            if not isinstance(image, (np.ndarray,)):
                raise TypeError("El tipo de dato de la imagen no es válido. Se esperaba un numpy.ndarray.")
            
           #comprobar que la imagen esté en escala de grises (2 canales) o en color (3 canales)
            if len(image.shape) < 2 or len(image.shape) > 3:
                raise ValueError("La imagen proporcionada tiene un número de dimensiones incorrecto.")
            
            img_gray = cv2.GaussianBlur(image, (7, 7), 0)
            return img_gray
        except cv2.error as e:
            print(f"Error de OpenCV durante el desenfoque de la imagen: {e}")
            raise
        except ValueError as ve:
            print(f"Valor incorrecto durante el desenfoque de la imagen: {ve}")
            raise
        except TypeError as te:
            print(f"Tipo incorrecto durante el desenfoque de la imagen: {te}")
            raise
        except Exception as ex:
            print(f"Ocurrió un error inesperado durante el desenfoque de la imagen: {ex}")
            raise
 
    def kernel_image(self, image):
        """
        Genera un kernel de convolución en forma de rectángulo para operaciones morfológicas.

        Args:
            image (np.ndarray): Imagen en formato NumPy array (aunque no se utiliza directamente en la función, 
                                se pasa como parte del proceso de pipeline).

        Returns:
            np.ndarray: Kernel de convolución para operaciones morfológicas.

        Raises:
            TypeError: Si la imagen no es un numpy.ndarray.
            Exception: Si ocurre un error inesperado durante la creación del kernel.
        """
        try:
            # Verificar que la imagen sea un np.ndarray, aunque no se use en esta función
            if not isinstance(image, np.ndarray):
                raise TypeError("La imagen proporcionada debe ser un numpy.ndarray.")
            
            # Crear un kernel de convolución de 5x5 en forma de rectángulo
            img_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            return img_k

        except TypeError as te:
            print(f"Error de tipo: {te}")
            raise

        except Exception as ex:
            print(f"Error inesperado durante la creación del kernel: {ex}")
            raise
    
    def dilate_image(self, image, kernel):
        """
        Aplica la dilatación a la imagen utilizando un kernel específico.

        Args:
            image (np.ndarray): Imagen binarizada en formato NumPy array a la que se le aplicará la dilatación.
            kernel (np.ndarray): Kernel de convolución para la operación de dilatación.

        Returns:
            np.ndarray: Imagen dilatada.

        Raises:
            ValueError: Si la imagen o el kernel es None.
            TypeError: Si la imagen o el kernel no son numpy.ndarray.
            cv2.error: Si ocurre un error con OpenCV durante la operación de dilatación.
            Exception: Para cualquier otro error inesperado.
        """
        try:
            # Verificar si la imagen y el kernel son válidos
            if image is None or kernel is None:
                raise ValueError("La imagen o el kernel proporcionados son None.")
            if not isinstance(image, np.ndarray) or not isinstance(kernel, np.ndarray):
                raise TypeError("Tanto la imagen como el kernel deben ser numpy.ndarray.")

            # Aplicar la operación de dilatación
            img_dilate = cv2.dilate(image, kernel, iterations=4)
            return img_dilate

        except ValueError as ve:
            print(f"Error de valor: {ve}")
            raise

        except TypeError as te:
            print(f"Error de tipo: {te}")
            raise

        except cv2.error as e:
            print(f"Error de OpenCV durante la dilatación: {e}")
            raise

        except Exception as ex:
            print(f"Error inesperado durante la dilatación: {ex}")
            raise
    
    def correct_illumination(self, image):
        """
        Corrige las variaciones de iluminación en la imagen utilizando la técnica de sustracción de fondo.

        Args:
            image (np.ndarray): Imagen en escala de grises.

        Returns:
            np.ndarray: Imagen con la iluminación corregida.
        """
        try:
            # Aplicar filtro morfológico para obtener la imagen de fondo
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            background = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

            # Restar el fondo de la imagen original
            corrected_image = cv2.subtract(background, image)
            corrected_image = cv2.normalize(corrected_image, None, 0, 255, cv2.NORM_MINMAX)
            return corrected_image

        except Exception as e:
            print(f"Error al corregir la iluminación: {e}")
            raise

    def adaptive_threshold(self, image):
        """
        Aplica umbralización que se adapta a la imagen proporcionada

        Args:
            image (np.ndarray): Imagen en escala de grises.

        Returns:
            np.ndarray: Imagen binarizada usando umbralización adaptativa.
        """
        try:
            adaptive_thresh = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
            )
            return adaptive_thresh

        except Exception as e:
            print(f"Error al aplicar umbralización adaptativa: {e}")
            raise

    def correccion_inclinacion(self, image):
        """
        Corrige la inclinación de la imagen utilizando.

        Args:
            image (np.ndarray): Imagen binarizada.

        Returns:
            np.ndarray: Imagen corregida sin inclinación.
        """
        try:
            coords = np.column_stack(np.where(image > 0))
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            print(f"Ángulo de rotación: {angle:.3f} grados")
            return rotated

        except Exception as e:
            print(f"Error al corregir la inclinación: {e}")
            raise

    def aplicar_nitidez(self, image):
        """
        Aplica un filtro para mejorar la nitidez de la imagen.

        Args:
            image (np.ndarray): Imagen en escala de grises.

        Returns:
            np.ndarray: Imagen con mayor nitidez.
        """
        try:
            kernel = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])
            sharpened = cv2.filter2D(image, -1, kernel)
            return sharpened

        except Exception as e:
            print(f"Error al aplicar filtro de nitidez: {e}")
            raise

    def remove_noise(self, image):
        """
        Elimina pequeños puntos o ruido en la imagen mediante operaciones morfológicas. Estas operaciones aplican erosion
        y dilatación para eliminar el ruido.

        Args:
            image (np.ndarray): Imagen binarizada.

        Returns:
            np.ndarray: Imagen sin ruido.
        """
        try:
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
            return opening

        except Exception as e:
            print(f"Error al eliminar ruido: {e}")
            raise

    def find_contours_hierarchy(self, image):
        """
        Busca todos los contornos en una imagen binarizada, incluyendo sus jerarquías.

        Args:
            image (np.ndarray): Imagen binarizada en formato NumPy array.

        Returns:
            list: Lista de contornos encontrados en la imagen.
            ndarray: Jerarquía de los contornos.

        Raises:
            ValueError: Si la imagen es None, no está en escala de grises o es inválida.
            TypeError: Si la imagen no es un numpy.ndarray.
            cv2.error: Si ocurre un error con OpenCV durante la búsqueda de contornos.
            Exception: Para cualquier otro error inesperado.
        """
        try:
            if image is None:
                raise ValueError("La imagen proporcionada no existe.")
            if not isinstance(image, (np.ndarray,)):
                raise TypeError("El tipo de dato de la imagen no es válido. Se esperaba un numpy.ndarray.")
            
            if len(image.shape) != 2:
                raise ValueError("La imagen proporcionada no está en escala de grises.")
            
            # Usar RETR_TREE para obtener todos los contornos y su jerarquía
            contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            return contours, hierarchy
        except cv2.error as e:
            print(f"Error de OpenCV durante la búsqueda de contornos: {e}")
            raise
        except ValueError as ve:
            print(f"Valor incorrecto durante la búsqueda de contornos: {ve}")
            raise
        except TypeError as te:
            print(f"Tipo incorrecto durante la búsqueda de contornos: {te}")
            raise
        except Exception as ex:
            print(f"Ocurrió un error durante la búsqueda de contornos: {ex}")
            raise


    #filtra los contornos para extraer los parrafos correcto teniendo en cuenta, tamaño y posicion en la imagen
    def filter_paragraph_contours(self, image, contours, min_x=150, min_y=80):
        """
        Filtra los contornos de una imagen para extraer aquellos que probablemente correspondan a párrafos,
        basándose en su tamaño y posición dentro de la imagen.

        Args:
            image (np.ndarray or Image.Image): Imagen en la que se detectaron los contornos.
            contours (list): Lista de contornos detectados en la imagen.
            min_x (int, optional): Ancho mínimo que debe tener un contorno para ser considerado válido. Valor por defecto es 150.
            min_y (int, optional): Alto mínimo que debe tener un contorno para ser considerado válido. Valor por defecto es 80.

        Returns:
            list: Lista de contornos válidos, donde cada contorno es una tupla (x, y, w, h) que indica la posición y tamaño del contorno.

        Raises:
            TypeError: Si la imagen no es un numpy.ndarray o una imagen PIL.
            ValueError: Si las dimensiones de la imagen no son válidas.
            Exception: Para cualquier otro error inesperado.
        """
        try:
            # Convertir la imagen a NumPy array si es una imagen PIL
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Verificar si la imagen es válida
            if not isinstance(image, np.ndarray):
                raise TypeError("La imagen debe ser un numpy.ndarray o PIL.Image.")

            # Verificar si la imagen tiene las dimensiones correctas
            if len(image.shape) < 2:
                raise ValueError("La imagen proporcionada tiene dimensiones inválidas.")

            # Obtener las dimensiones de la imagen
            image_h, image_w = image.shape[:2]
            margen_x_left = 0.10 * image_w
            margen_x_right = 0.90 * image_w
            margen_y_top = 0.05 * image_h
            margen_y_bot = 0.95 * image_h
            
            valid_contours = []

            # Filtrar contornos por tamaño y posición en la imagen
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # Verificar si el contorno está dentro de los márgenes
                if x > margen_x_left and (x + w) < margen_x_right and y > margen_y_top and (y + h) < margen_y_bot:
                    # Verificar si el contorno tiene un tamaño mínimo
                    if w > min_x and h > min_y:
                        valid_contours.append((x, y, w, h))
                        print(f"Contorno válido: ({x}, {y}), ancho: {w}, alto: {h}")
                    else:
                        print(f"Contorno descartado por tamaño insuficiente (w={w}, h={h}).")
                else:
                    print(f"Contorno descartado por no estar en la zona central.")

            return valid_contours

        except TypeError as te:
            print(f"Error de tipo: {te}")
            raise

        except ValueError as ve:
            print(f"Error de valor: {ve}")
            raise

        except Exception as ex:
            print(f"Error inesperado: {ex}")
            raise

    #funcion para añadir contornos a la imagen del documento
    #retorna la imagen además de las posiciones de los contornos para su posterior uso (quizás sea mejor separar ambas funcionalidades)
    def extract_image_with_contours(self, images):
        """
        Extrae y dibuja los contornos de párrafos en cada imagen del documento.

        Args:
            images (dict): Diccionario de imágenes en formato NumPy array, donde las claves son los números de página.

        Returns:
            tuple:
                dict: Diccionario de imágenes con contornos dibujados, donde las claves son los números de página.
                dict: Diccionario de posiciones de contornos, donde cada clave es el número de página, y los valores son 
                    listas de posiciones de los contornos ({'x', 'y', 'width', 'height'}).

        Raises:
            TypeError: Si alguno de los valores de `images` no es un numpy.ndarray.
            ValueError: Si las imágenes están vacías o no pueden ser procesadas.
            cv2.error: Si ocurre un error con OpenCV durante el procesamiento.
            Exception: Para cualquier otro error inesperado.
        """
        try:
            if not images:
                raise ValueError("El diccionario de imágenes no puede estar vacío")
            
            images_with_contours = {}
            contours_positions = {}

            for page_num, image in images.items():
                # Verificar que la imagen es un numpy.ndarray
                if not isinstance(image, np.ndarray):
                    raise TypeError(f"La imagen en la página {page_num} no es un numpy.ndarray.")

                # Convertir la imagen a escala de grises
                gray_img = self.convert_to_gray(image)

                # Aplicar desenfoque y umbralización
                blur_img = self.blur_image(gray_img)
                thresh_img = self.thresh_otsu(blur_img)

                # Invertir la imagen binaria
                inv_img = self.invert_image(thresh_img)

                # Crear el kernel para la operación de dilatación
                kernel = self.kernel_image(inv_img)

                # Aplicar dilatación a la imagen
                dilate = self.dilate_image(inv_img, kernel)

                # Buscar los contornos en la imagen dilatada
                contours = self.search_contours(dilate)

                # Filtrar los contornos que probablemente correspondan a párrafos
                valid_contours = self.filter_paragraph_contours(image, contours)

                # Si se encuentran contornos válidos, dibujarlos y guardar posiciones
                if valid_contours:
                    image_with_rects = np.array(image)
                    positions = []

                    for x, y, w, h in valid_contours:
                        cv2.rectangle(image_with_rects, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        positions.append({'x': x, 'y': y, 'width': w, 'height': h})

                    images_with_contours[page_num] = image_with_rects
                    contours_positions[page_num] = positions
                    print(f"Imágenes y párrafos procesados con éxito en la página {page_num}")
                else:
                    print(f"No se encontraron contornos válidos en la página {page_num}.")

            return images_with_contours, contours_positions

        except TypeError as te:
            print(f"Error de tipo: {te}")
            raise

        except ValueError as ve:
            print(f"Error de valor: {ve}")
            raise

        except cv2.error as e:
            print(f"Error de OpenCV durante el procesamiento de la imagen: {e}")
            raise

        except Exception as ex:
            print(f"Error inesperado durante el procesamiento de la imagen: {ex}")
            raise

    def filter_title_contours(self, image, contours, min_width_ratio=0.12, max_height_ratio=0.07, density_threshold=0.25):
        """
        Filtra los contornos que corresponden a títulos según su tamaño relativo y densidad de píxeles negros.
        Se ha ajustado el ratio de ancho y altura para detectar mejor los títulos.
        
        Args:
            image: La imagen original en la que se buscan los títulos.
            contours: Los contornos detectados en la imagen procesada.
            min_width_ratio: Relación mínima de ancho en función del ancho total de la imagen.
            max_height_ratio: Relación máxima de altura en función de la altura total de la imagen.
            density_threshold: Umbral de densidad de píxeles negros para identificar títulos.
        
        Returns:
            valid_titles: Lista de contornos que corresponden a títulos válidos.
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        image_h, image_w = image.shape[:2]
        print(f"Altura de la imagen: {image_h}, Ancho de la imagen: {image_w}")

        valid_titles = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            print(f"Contorno encontrado en ({x}, {y}), ancho: {w}, alto: {h}")
            
            if w > min_width_ratio * image_w and h < max_height_ratio * image_h:
                # Extraer la región del contorno
                region_contour = image[y:y+h, x:x+w]
                # Calcular la densidad de píxeles negros (0 representa negro en imágenes binarizadas)
                pixel_density = np.sum(region_contour == 0) / (w * h)
                print(f"Densidad de píxeles: {pixel_density}")
                
                # Filtrar por densidad de píxeles negros
                if pixel_density > density_threshold:
                    valid_titles.append((x, y, w, h))
                    print(f"Contorno válido como título: ({x}, {y}), ancho: {w}, alto: {h}")
                else:
                    print(f"Contorno descartado por baja densidad de píxeles: {pixel_density}")
            else:
                print(f"Contorno descartado por no cumplir con las medidas relativas (w={w}, h={h}).")
                
        return valid_titles
    
    def extraer_titulos(self, images):
        """
        Procesa cada página de la lista de imágenes para extraer las posiciones de los títulos mediante
        técnicas de procesamiento de imágenes y detección de contornos.

        Args:
            images (dict): Diccionario donde las claves son los números de página y los valores son las imágenes 
                        en formato NumPy array.

        Returns:
            tuple:
                dict: Diccionario de imágenes con los contornos de los títulos dibujados.
                dict: Diccionario de posiciones de los títulos ({'x', 'y', 'width', 'height'}) por cada página.

        Raises:
            TypeError: Si alguna imagen no es un numpy.ndarray.
            ValueError: Si las imágenes están vacías o no pueden ser procesadas.
            cv2.error: Si ocurre un error con OpenCV durante el procesamiento de la imagen.
            Exception: Para cualquier otro error inesperado.
        """
        try:
            if not images:
                raise ValueError("El diccionario de imágenes está vacío.")
            image_with_titles = {}
            titles_position = {}

            # Procesar cada imagen en el diccionario
            for page_num, image in images.items():
                # Verificar que la imagen es un numpy.ndarray
                if isinstance(image, Image.Image):
                    image = np.array(image)
                elif not isinstance(image, np.ndarray):
                    raise TypeError(f"La imagen en la página {page_num} no es un numpy.ndarray ni una PIL.Image.")

                # Convertir la imagen a escala de grises
                gray_img = self.convert_to_gray(image)

                # Aplicar desenfoque y umbralización Otsu
                blur_img = self.blur_image(gray_img)
                thresh_img = self.thresh_otsu(blur_img)

                # Invertir la imagen binaria
                inv_img = self.invert_image(thresh_img)

                # Crear kernel para dilatación y aplicar dilatación a la imagen
                kernel = self.kernel_image(inv_img)
                dilate = self.dilate_image(inv_img, kernel)

                # Buscar contornos en la imagen dilatada
                contours = self.search_contours(dilate)
                
                # Filtrar los contornos que corresponden a títulos válidos
                valid_titles = self.filter_title_contours(image, contours)

                # Si se encuentran títulos válidos, dibujarlos y guardar posiciones
                if valid_titles:
                    image_with_titles_page = np.array(image)
                    positions = []

                    # Dibujar los rectángulos en los contornos de títulos
                    for x, y, w, h in valid_titles:
                        cv2.rectangle(image_with_titles_page, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        positions.append({'x': x, 'y': y, 'width': w, 'height': h})

                    # Guardar las imágenes y las posiciones de los títulos
                    image_with_titles[page_num] = image_with_titles_page
                    titles_position[page_num] = positions
                    print(f"Títulos extraídos con éxito en la página {page_num}")
                else:
                    print(f"No se encontraron contornos válidos en la página {page_num}.")
            
            print(titles_position)
            return image_with_titles, titles_position

        except TypeError as te:
            print(f"Error de tipo: {te}")
            raise

        except ValueError as ve:
            print(f"Error de valor: {ve}")
            raise

        except cv2.error as e:
            print(f"Error de OpenCV durante el procesamiento de la imagen: {e}")
            raise

        except Exception as ex:
            print(f"Error inesperado durante el procesamiento de la imagen: {ex}")
            raise
    
    def process_document(self, images):
        """
        Procesa cada página de la lista de imágenes. Primero extrae los títulos y luego añade los contornos
        de los párrafos a las imágenes ya procesadas.

        Args:
            images (dict): Diccionario de imágenes en formato NumPy array, donde las claves son los números de página.

        Returns:
            tuple:
                dict: Diccionario de imágenes con los contornos de párrafos y títulos dibujados.
                dict: Diccionario de posiciones de los títulos.
                dict: Diccionario de posiciones de los párrafos.

        Raises:
            TypeError: Si alguna imagen no es un numpy.ndarray.
            ValueError: Si las imágenes están vacías o no pueden ser procesadas.
            cv2.error: Si ocurre un error con OpenCV durante el procesamiento de la imagen.
            Exception: Para cualquier otro error inesperado.
        """
        try:
            if not images:
                raise ValueError("El diccionario de imágenes está vacío.")
            print("Procesando títulos en las imágenes...")
            images_with_titles, titles_position = self.extraer_titulos(images)

            print("Procesando párrafos en las imágenes con contornos de títulos...")
            images_with_paragraphs, paragraphs_position = self.extract_image_with_contours(images_with_titles)

            for page_num, image_with_paragraphs in images_with_paragraphs.items():
                image_name = f'pagina_{page_num}_procesada'
                

            return images_with_paragraphs, titles_position, paragraphs_position

        except TypeError as te:
            print(f"Error de tipo: {te}")
            raise

        except ValueError as ve:
            print(f"Error de valor: {ve}")
            raise

        except cv2.error as e:
            print(f"Error de OpenCV durante el procesamiento de la imagen: {e}")
            raise

        except Exception as ex:
            print(f"Error inesperado durante el procesamiento del documento: {ex}")
            raise
    
    def extraer_bloques_secuenciales(self, images, titles_position, paragraphs_position):
        """
        Extrae los bloques de texto (títulos y párrafos) de todas las imágenes del documento de manera secuencial,
        sin preocuparse por en qué página están. Los bloques se añaden en el orden correcto de aparición.

        Args:
            images (dict): Diccionario de imágenes en formato NumPy array o PIL Image.
            titles_position (dict): Diccionario con las posiciones de los contornos de los títulos.
            paragraphs_position (dict): Diccionario con las posiciones de los contornos de los párrafos.

        Returns:
            list: Lista de bloques de texto extraídos en orden, donde cada bloque contiene tipo ('título' o 'párrafo'),
                posición (x, y, ancho, alto), el texto extraído mediante OCR, y la imagen del bloque.

        Raises:
            TypeError: Si las imágenes no son numpy.ndarray o PIL.Image.
            ValueError: Si los datos de las posiciones no son válidos.
            Exception: Para cualquier otro error inesperado.
        """
        try:
            bloques = []
            last_page_num = max(images.keys())
            # Iterar sobre todas las páginas en el orden correcto
            for page_num, image in images.items():
                print(f"PROCESANDO LA PÁGINA: {page_num}")

                # Convertir la imagen a NumPy array si es PIL Image
                if isinstance(image, Image.Image):
                    image = np.array(image)
                elif not isinstance(image, np.ndarray):
                    raise TypeError(f"La imagen en la página {page_num} no es un numpy.ndarray ni una PIL.Image.")

                self.save_single_image(image, f"PAGE_NUM {page_num}")

                # Combinar los títulos y párrafos en una lista ordenada según su posición en la página
                bloques_pagina = []

                # Añadir los títulos
                for title_position in titles_position.get(page_num, []):
                    x, y, w, h = title_position['x'], title_position['y'], title_position['width'], title_position['height']
                    title_region = image[y:y+h, x:x+w]  # Extraer la región del título
                    texto_titulo = ocr.ocr_to_img(title_region)  # Aplicar OCR y limpiar
                    bloque_titulo = {
                        'tipo': 'título',
                        'posicion': (x, y, w, h),
                        'texto': texto_titulo,
                        'imagen': title_region
                    }
                    bloques_pagina.append(bloque_titulo)  # Añadir el bloque a la lista

                # Añadir los párrafos
                for paragraph_position in paragraphs_position.get(page_num, []):
                    x, y, w, h = paragraph_position['x'], paragraph_position['y'], paragraph_position['width'], paragraph_position['height']
                    paragraph_region = image[y:y+h, x:x+w]  # Extraer la región del párrafo
                    texto_parrafo = ocr.ocr_to_img(paragraph_region)  # Aplicar OCR y limpiar
                    bloque_parrafo = {
                        'tipo': 'párrafo',
                        'posicion': (x, y, w, h),
                        'texto': texto_parrafo,
                        'imagen': paragraph_region
                    }
                    bloques_pagina.append(bloque_parrafo)  # Añadir el bloque a la lista
                    

                # Ordenar los bloques de la página por su posición vertical (coordenada y)
                bloques_pagina = sorted(bloques_pagina, key=lambda bloque: bloque['posicion'][1])
                

                # Añadir los bloques de la página a la lista general
                bloques.extend(bloques_pagina)
                if page_num == last_page_num:
                    print(f"\nBLOQUES DE LA ÚLTIMA PÁGINA ({page_num}):")
                    for bloque in bloques_pagina:
                        print(bloque)

            return bloques

        except TypeError as te:
            print(f"Error de tipo: {te}")
            raise

        except ValueError as ve:
            print(f"Error de valor: {ve}")
            raise

        except Exception as ex:
            print(f"Error inesperado durante la extracción de bloques secuenciales: {ex}")
            raise

    def is_column_non_empty(self, image, threshold=10):
        """
        Checks if the column image contains text by analyzing the pixel values.
        
        Args:
            image (np.ndarray): The column image to check.
            threshold (int): Minimum number of non-zero pixels to consider the column as non-empty.
        
        Returns:
            bool: True if the column is non-empty, False otherwise.
        """
        try:
            # Convert to grayscale
            gray_image = self.convert_to_gray(image)
            # Apply thresholding to binarize the image
            _, thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Invert the image to make text white on black background
            inverted_image = cv2.bitwise_not(thresh_image)
            # Count non-zero pixels (which represent text)
            non_zero_pixels = cv2.countNonZero(inverted_image)
            if non_zero_pixels > threshold:
                return True
            else:
                return False
        except Exception as ex:
            print(f"Error checking if column is non-empty: {ex}")
            return False

    def extract_column_images(self, images):
        """
        Extracts the column images from the provided images, skipping empty columns.

        Args:
            images (dict): Dictionary of images in NumPy array format, where keys are the page numbers.

        Returns:
            dict: Dictionary of non-empty column images, with keys as (page_num, 'left' or 'right').
        """
        column_images = {}
        for page_num, image in images.items():
            if isinstance(image, Image.Image):
                image = np.array(image)
            elif not isinstance(image, np.ndarray):
                raise TypeError(f"La imagen en la página {page_num} no es un numpy.ndarray ni una PIL.Image.")

            height, width = image.shape[:2]

            # Split into left and right columns
            left_column = image[:, :width // 2]
            right_column = image[:, width // 2:]

            # Check if left column contains text
            if self.is_column_non_empty(left_column):
                column_images[(page_num, 'left')] = left_column
                print(f"La columna izquierda en la página {page_num} contiene texto y será procesada.")
            else:
                print(f"La columna izquierda en la página {page_num} está vacía y será omitida.")

            # Check if right column contains text
            if self.is_column_non_empty(right_column):
                column_images[(page_num, 'right')] = right_column
                print(f"La columna derecha en la página {page_num} contiene texto y será procesada.")
            else:
                print(f"La columna derecha en la página {page_num} está vacía y será omitida.")

        return column_images

    
    def filter_column_title_contours(self, image, contours, min_width_ratio=0.05, max_height_ratio=0.05, density_threshold=0.10):
        """
        Filtra los contornos que corresponden a títulos según su tamaño relativo y densidad de píxeles negros.
        Se ha ajustado el ratio de ancho y altura para detectar mejor los títulos.
        
        Args:
            image: La imagen original en la que se buscan los títulos.
            contours: Los contornos detectados en la imagen procesada.
            min_width_ratio: Relación mínima de ancho en función del ancho total de la imagen.
            max_height_ratio: Relación máxima de altura en función de la altura total de la imagen.
            density_threshold: Umbral de densidad de píxeles negros para identificar títulos.
        
        Returns:
            valid_titles: Lista de contornos que corresponden a títulos válidos.
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        image_h, image_w = image.shape[:2]
        print(f"Altura de la imagen: {image_h}, Ancho de la imagen: {image_w}")

        valid_titles = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            print(f"Contorno encontrado en ({x}, {y}), ancho: {w}, alto: {h}")
            
            if w > min_width_ratio * image_w and h < max_height_ratio * image_h:
                # Extraer la región del contorno
                region_contour = image[y:y+h, x:x+w]
                # Calcular la densidad de píxeles negros (0 representa negro en imágenes binarizadas)
                pixel_density = np.sum(region_contour == 0) / (w * h)
                print(f"Densidad de píxeles: {pixel_density}")
                
                # Filtrar por densidad de píxeles negros
                if pixel_density > density_threshold:
                    valid_titles.append((x, y, w, h))
                    print(f"Contorno válido como título: ({x}, {y}), ancho: {w}, alto: {h}")
                else:
                    print(f"Contorno descartado por baja densidad de píxeles: {pixel_density}")
            else:
                print(f"Contorno descartado por no cumplir con las medidas relativas (w={w}, h={h}).")
                
        return valid_titles
    
    def extract_title_positions(self, images):
        """
        Extracts the positions of the titles in the provided images.

        Args:
            images (dict): Dictionary of images (column images), where keys are tuples like (page_num, 'left' or 'right').

        Returns:
            dict: Dictionary of title positions for each column image.
        """
        try:
            if not images:
                raise ValueError("El diccionario de imágenes está vacío.")
            
            titles_position = {}

            for (page_num, column_side), image in images.items():
                # Verify the image is a numpy.ndarray
                if isinstance(image, Image.Image):
                    image = np.array(image)
                elif not isinstance(image, np.ndarray):
                    raise TypeError(f"La imagen en la página {page_num}, columna {column_side} no es un numpy.ndarray ni una PIL.Image.")

                # Convertir la imagen a escala de grises
                gray_img = self.convert_to_gray(image)

                # Aplicar desenfoque y umbralización Otsu
                blur_img = self.blur_image(gray_img)
                thresh_img = self.thresh_otsu(blur_img)

                # Invertir la imagen binaria
                inv_img = self.invert_image(thresh_img)

                # Crear kernel para dilatación y aplicar dilatación a la imagen
                kernel = self.kernel_image(inv_img)
                dilate = self.dilate_image(inv_img, kernel)

                # Buscar contornos en la imagen dilatada
                contours = self.search_contours(dilate)
                
                # Filtrar los contornos que corresponden a títulos válidos
                valid_titles = self.filter_column_title_contours(image, contours)

                # Guardar las posiciones de los títulos
                if valid_titles:
                    positions = [{'x': x, 'y': y, 'width': w, 'height': h} for x, y, w, h in valid_titles]
                    titles_position[page_num] = positions
                    print(f"Títulos extraídos con éxito en la página {page_num}")

                    # Dibujar los rectángulos en los contornos de títulos
                    for x, y, w, h in valid_titles:
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Guardar la imagen con los contornos dibujados
                    self.save_single_image(image, f"titles_page_{page_num}.png")
                else:
                    print(f"No se encontraron contornos válidos en la página {page_num}.")
            
            return titles_position

        except TypeError as te:
            print(f"Error de tipo: {te}")
            raise

        except ValueError as ve:
            print(f"Error de valor: {ve}")
            raise

        except cv2.error as e:
            print(f"Error de OpenCV durante el procesamiento de la imagen: {e}")
            raise

        except Exception as ex:
            print(f"Error inesperado durante la extracción de posiciones de títulos: {ex}")
            raise

    
    def draw_rectangles_on_titles(self, column_images, titles_position):
        """
        Draws rectangles around the titles in the column images.

        Args:
            column_images (dict): Dictionary of column images in NumPy array format.
            titles_position (dict): Dictionary with positions of the title contours.

        Returns:
            list: List of images with rectangles drawn around the titles.
        """
        images_with_rectangles = []

        for (page_num, column_side), image in column_images.items():
            # Verify that the image is a numpy.ndarray
            if not isinstance(image, np.ndarray):
                raise TypeError(f"La imagen en la columna {column_side} de la página {page_num} no es un numpy.ndarray.")

            # Create a copy of the image to draw rectangles
            image_with_rects = np.array(image)

            # Get title positions for this column if any
            positions = titles_position.get((page_num, column_side), [])

            # Draw rectangles around titles
            for title_position in positions:
                x = title_position['x']
                y = title_position['y']
                w = title_position['width']
                h = title_position['height']
                cv2.rectangle(image_with_rects, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Optionally save or process the image
            self.save_single_image(image_with_rects, f"titles_page_{page_num}_{column_side}")

            images_with_rectangles.append(image_with_rects)

        return images_with_rectangles



if __name__ == '__main__':
    pti = ProcessToImages()
    ocr = OCRProducer()
    doc1 = './docs_pruebas/doc_scan2.pdf'
    images = pti.convert_pdf_to_image_from_file(doc1)
    """image_parrafos, positions = pti.extract_image_with_contours(images)
    images, titulos = pti.extraer_titulos(images)
    titulos_y_parrafos_images = pti.extraer_texto_titulos(images, titulos)
    images, t_pos, p_pos = pti.process_document(images)
    bloques = pti.extraer_bloques_secuenciales(images, t_pos, p_pos)"""
    column_images = pti.extract_column_images(images)
    titles_position = pti.extract_title_positions(column_images)
    pti.draw_rectangles_on_titles(column_images, titles_position)
    
    

