import logging
from pdf2image import convert_from_bytes, convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError
import os

#Clase que convierte los pdf a imágenes y las guarda
class PDFConverter:
    def __init__(self, output_folder = 'output_images'):
       
        self.output_folder = output_folder
        
        #se crea el directorio si no existe
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    #Función que convierte y retorna los pdf a una lista de imágenes verificando previamente si su extensión es .pdf. 
    def convert_to_images(self, pdf_file):

        try:
            images = convert_from_bytes(pdf_file)
            logging.debug(f"Successfully converted PDF to images: {images}")
        except FileNotFoundError as e:
            logging.error(f"FileNotFoundError: {e}")
            return []
        except OSError as e:
            logging.error(f"OSError: {e}")
            return []
        except PDFInfoNotInstalledError as e:
            logging.error(f"PDFInfoNotInstalledError: {e}")
            return []
        except PDFPageCountError as e:
            logging.error(f"PDFPageCountError: {e}")
            return []
        except PDFSyntaxError as e:
            logging.error(f"PDFSyntaxError: {e}")
            return []
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return []
        
        return images
    
    def convert_to_images_from_file(self, pdf_file):

        try:
            images = convert_from_path(pdf_file)
            logging.debug(f"Successfully converted PDF to images: {images}")
        except FileNotFoundError as e:
            logging.error(f"FileNotFoundError: {e}")
            return []
        except OSError as e:
            logging.error(f"OSError: {e}")
            return []
        except PDFInfoNotInstalledError as e:
            logging.error(f"PDFInfoNotInstalledError: {e}")
            return []
        except PDFPageCountError as e:
            logging.error(f"PDFPageCountError: {e}")
            return []
        except PDFSyntaxError as e:
            logging.error(f"PDFSyntaxError: {e}")
            return []
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return []
        
        return images

    #Guardamos las imágenes en una carpeta usando el nombre del archivo
    def save_images(self, route_path):
        for i, image in enumerate(self.images):
            slash = route_path.rfind('/')
            punto = route_path.rfind('.')
            if slash != -1 and punto != -1 and slash < punto:
                file_name = route_path[slash + 1:punto]
                image_path = os.path.join(self.output_folder, f'{file_name}_{i+1}.png')
            image.save(image_path, 'PNG')


    

