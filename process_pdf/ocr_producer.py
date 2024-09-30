import logging
import cv2
import numpy as np
import pytesseract
import os
from PIL import Image

#Clase que usa pytesseract para OCR
class OCRProducer:
    def __init__(self):
        #Incluir lo siguiente si no está habilitado la ruta completa de pytesseract al PATH
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
       
        
    #Función que procesa la lista de imágenes de los pdf y los convierte en texto. 
    # Cada texto se pasa como elemento a una lista
    def image_to_string_combined_text(self, images):
        
        result_text = []
        if not isinstance(images, list):
            raise ValueError("La entrada de datos no corresponde a una lista de imágenes para su procesamiento")
            
        for image in images:
            try:
                text = pytesseract.image_to_string(image, lang='spa')
                result_text.append(text)
            except ValueError as e:
                print(f"Error: {e}")
            except pytesseract.TesseractError as te:
                logging.error(f"Error de Tesseract al procesar la imagen: {te}")
                result_text.append("")
            
        combined_text = ' '.join(result_text)
        
        return combined_text
    
    def image_to_string(self, images):
        result_text = []
        if not isinstance(images, list):
            raise ValueError("La entrada de datos no corresponde a una lista de imágenes para su procesamiento")
            
        for image in images:
            try:
                text = pytesseract.image_to_string(image, lang='spa')
                result_text.append(text)
            except ValueError as e:
                print(f"Error: {e}")
            except pytesseract.TesseractError as te:
                logging.error(f"Error de Tesseract al procesar la imagen: {te}")
                result_text.append("")

        return result_text
    

    def ocr_to_img(self, image):
        """
        Aplica OCR con pytesseract a una imagen en formato NumPy y devuelve el texto extraído.
        
        Args:
            image: Imagen en formato NumPy array o PIL.
        
        Returns:
            Texto extraído tras aplicar OCR.
        
        Raises:
            TypeError: Si la imagen no está en formato NumPy array.
            ValueError: Si la imagen está vacía o no es válida para el procesamiento de OCR.
            TesseractError: Si ocurre un error en el procesamiento de pytesseract.
        """
        try:
            # Verificar si la imagen es un NumPy array o PIL Image
            if not isinstance(image, (np.ndarray, Image.Image)):
                # Si la imagen no es ni NumPy array ni PIL Image, lanzar una excepción
                raise TypeError("La imagen debe estar en formato NumPy array o PIL Image para el procesamiento.")

            # Aplicar OCR usando pytesseract
            img_txt = pytesseract.image_to_string(image)

            return img_txt

        except TypeError as te:
            print(f"Error de tipo: {str(te)}")
            raise

        except ValueError as ve:
            print(f"Error de valor: {str(ve)}")
            raise

        except pytesseract.TesseractError as te:
            print(f"Error de Tesseract: {str(te)}")
            raise

        except Exception as e:
            print(f"Error inesperado: {str(e)}")
            raise

    

    


