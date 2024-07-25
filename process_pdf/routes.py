import logging
import os
from flask import Flask, jsonify, request, Blueprint
from encoder import Encoder
from pdf_converter import PDFConverter
from ocr_producer import OCRProducer

# Configurar que el logging se guarde en un archivo
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[logging.FileHandler("app.log"),
                              logging.StreamHandler()])

analize = Blueprint('analize', __name__)
converter =  PDFConverter()
producer = OCRProducer()


@analize.route('/')
def view():
    return '<h1> ANALIZADOR DE TEXTOS<h1>'

@analize.route('/analizar_documentos', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        data = request.get_json()

        logging.debug(f"Datos recibidos en la solicitud post: {data}")
        if 'file' not in data or 'filename' not in data:
            return jsonify("Error con el archivo. No se han encontrado los datos del archivo")
        
        filename = data['filename']
        encoded_file = data['file']

        if not filename.lower().endswith('.pdf'):
            return jsonify("El archivo debe ser un PDF")
        
        try:
            decoded_file = Encoder.decode_file(encoded_file)
            logging.debug("Archivo decodificado correctamente.")

            with open("temp.pdf", "wb") as temp_pdf:
                temp_pdf.write(decoded_file)

            if not Encoder.validate_pdf(decoded_file):
                return jsonify({"error": "El archivo decodificado no parece ser un archivo PDF válido."}), 400

            os.remove("temp.pdf")
        except Exception as e:
            return jsonify({"error": f"Error al decodificar el archivo: {str(e)}"})

        try:
            logging.debug("Iniciando conversión a imágenes.")
            process = converter.convert_to_images(decoded_file)
            logging.debug(f"Resultado de la conversión a imágenes: {process}")
            texts = producer.process_images(process)
            logging.debug(f"Resultado del procesamiento de imágenes a textos. {texts}")
        except Exception as e:
            return jsonify({"error": f"Error en la conversión del archivo: {str(e)}"})

        return jsonify(f"Subida y procesamiento de archivos completada nombre: {filename}, resultado: {len(process)} imágenes\n"
                       f"Procesamiento de imágenes completado. Resultado: {texts}")


    return jsonify("Método no soportado")




