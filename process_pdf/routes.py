import logging
from flask import Flask, jsonify, request, Blueprint
from encoder import Encoder
from pdf_converter import PDFConverter
from ocr_producer import OCRProducer

# Configurar que el logging se guarde en un archivo
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[logging.FileHandler("app.log"),
                              logging.StreamHandler()])

file_path = r'C:/Users/Nel/Desktop/vectorizacion_2.pdf'
analize = Blueprint('analize', __name__)
encoder = Encoder(file_path)
converter = PDFConverter(file_path)
producer = OCRProducer()


@analize.route('/')
def view():
    return '<h1> ANALIZADOR DE TEXTOS<h1>'

@analize.route('/analizar_documentos', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        data = request.get_json()
        logging.debug(f"Datos recibidos en la solicitud post: {data}")
        if 'file' not in data or 'filename' not in data:
            return jsonify("Error con el archivo. No se ha encontrado el archivo y/o el nombre")

        encoded_file = data['file']
        filename = data['filename']
        logging.debug(f"Nombre del archivo: {filename}")

         # Verificar que el archivo tiene extensión .pdf
        if not filename.lower().endswith('.pdf'):
            return jsonify("El archivo debe ser un PDF")

        try:
            decoded_file = Encoder.decode_file(encoded_file)  
            logging.debug("Archivo decodificado correctamente.")
        except Exception as e:
            return jsonify({"error": f"Error al decodificar el archivo: {str(e)}"})
        
        #Verificar que los primeros caracteres correspondan a un archivo pdf
        if decoded_file[:4] != b'%PDF':
            return jsonify({"error": "El archivo decodificado no parece ser un archivo PDF válido."}), 400
        
        try:
            logging.debug("Iniciando conversión a imágenes.")
            process = converter.convert_to_images(decoded_file)
            logging.debug(f"Resultado de la conversión a imágenes: {process}")
        except Exception as e:
            return jsonify({"error": f"Error en la conversión del archivo: {str(e)}"})
        
        str_file = producer.process_images(process)

        return jsonify(f"Subida y procesamiento de archivos completada nombre: {filename}, resultado: {str_file}")

    return jsonify("Método no soportado")



