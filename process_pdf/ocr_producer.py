import logging
import pytesseract
import os

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
    
    


