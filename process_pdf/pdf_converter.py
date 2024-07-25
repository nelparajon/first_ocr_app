import logging
from pdf2image import convert_from_bytes, convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError
import os

#Clase que convierte los pdf a imágenes y las guarda
class PDFConverter:
    def __init__(self, pdf_path, output_folder = 'output_images'):
        self.pdf_path = pdf_path
        self.output_folder = output_folder
        
        #se crea el directorio si no existe
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    #Función que convierte y retorna los pdf a una lista de imágenes verificando previamente si su extensión es .pdf. 
    def convert_to_images(self, route_path):
        """if not route_path.endswith('.pdf'):
            raise ValueError("El archivo debe tener una extensión .pdf") Se pasa esto al método POST que sube el archivo en routes"""

        try:
            self.images = convert_from_bytes(route_path)
        except FileNotFoundError as e:
            logging.error(f"Error al no encontrar el archivo en la ruta especificada {route_path}: {e}")
            return []
        except OSError as e:
            logging.error(f"Error al convertir el archivo.pdf a imágenes: {e}")
            return []
        except (PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError) as e:
            logging.error(f"Error al procesar el archivo PDF: {e}")
            return []
    
        return self.images
    
    #Guardamos las imágenes en una carpeta usando el nombre del archivo
    def save_images(self, route_path):
        for i, image in enumerate(self.images):
            slash = route_path.rfind('/')
            punto = route_path.rfind('.')
            if slash != -1 and punto != -1 and slash < punto:
                file_name = route_path[slash + 1:punto]
                image_path = os.path.join(self.output_folder, f'{file_name}_{i+1}.png')
            image.save(image_path, 'PNG')


    

