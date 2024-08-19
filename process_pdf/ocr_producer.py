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
    def process_images(self, images):
        
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
    
    #Para evitar que se acumulen textos, se elimina el contenido del directorio 
    def delete_files(self, output_folder='output_texts'):
        try:
            files = os.listdir(output_folder)
        except FileNotFoundError as e:
            logging.error(f"Error: La carpeta especificada {output_folder} no existe.")
            return
        
        for f in files:
            try:
                file_path = os.path.join(output_folder, f)
            
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"{f} en la carpeta {output_folder}: Eliminado")

            except FileNotFoundError as e:
                logging.error(f"Error: El archivo {file_path} no existe.")
            except OSError as e:
                logging.error(f"Error al eliminar el archivo {file_path}: {e}")


    #Se guardan los textos en un directorio
    def save_texts(self, texts, output_folder='output_texts'):
        
        try:
            os.makedirs(output_folder, exist_ok=True)
        except OSError as oe:
            logging.error(f"Error al crear el directorio {output_folder}: {oe}")
        
        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
            except OSError as e:
                logging.error(f"Error al crear el directorio {output_folder}: {e}")
                return
            
            self.delete_files(output_folder)

        for i, t in enumerate(texts):
            file_name = os.path.join(output_folder, f'pagina_{i+1}.txt')                
            try:
                with open(file_name, "w", encoding='utf-8') as file:
                    file.write(t)
            except IOError as e:
                logging.error(f"Error al escribir en el archivo {file_name}: {e}")
            except Exception as e:
                logging.error(f"Error inesperado al escribir en el archivo {file_name}: {e}")
                


