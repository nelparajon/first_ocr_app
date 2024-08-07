import logging
import os
from flask import Flask, jsonify, request, Blueprint
import numpy as np
from werkzeug.exceptions import BadRequest, InternalServerError
from cliente.encoder import Encoder
from process_pdf.pdf_converter import PDFConverter
from process_pdf.ocr_producer import OCRProducer
from process_pdf.tokenizer import Tokenizer
from process_pdf.lemmatizer import Lemmatizer
from process_pdf.vectorizer import Vectorizer
from database.db_manager import DbManager

#Configurar que el logging se guarde en un archivo
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[logging.FileHandler("app.log"),
                              logging.StreamHandler()])

analize = Blueprint('analize', __name__)
converter =  PDFConverter()
producer = OCRProducer()
tokenizer = Tokenizer()
lemmatizer = Lemmatizer()
vectorizer = Vectorizer()

analyze_service_route = os.getenv('ANALYZE_SERVICE_ROUTE')
service_get_historico = os.getenv('GET_HISTORICO')


@analize.route('/')
def view():
    return '<h1> ANALIZADOR DE TEXTOS<h1>'

@analize.route(analyze_service_route, methods=['POST'])
def analizar_documentos():
    if 'pdf1' not in request.files or 'pdf2' not in request.files:
        return jsonify({'error': 'No se encontraron los archivos'}), 400

    pdf1 = request.files['pdf1']
    pdf2 = request.files['pdf2']

    pdf1_filename = pdf1.filename
    pdf2_filename = pdf2.filename

    try:
        # Leer los archivos y convertirlos a base64
        pdf1_base64 = Encoder.encode_file_b64(pdf1)
        pdf2_base64 = Encoder.encode_file_b64(pdf2)

        # Decodificar los archivos base64
        decoded_pdf1 = Encoder.decode_file(pdf1_base64)
        decoded_pdf2 = Encoder.decode_file(pdf2_base64)

        # Validar que los archivos sean PDFs
        if not Encoder.validate_pdf(decoded_pdf1) or not Encoder.validate_pdf(decoded_pdf2):
            return handle_response(405, f"Subida y procesamiento de archivos completada.", f"{porcentaje:.2f}")

        # Se guardan los archivos decodificados de manera temporal para el análisis de errores
        with open("temp_1.pdf", "wb") as temp_pdf:
            temp_pdf.write(decoded_pdf1)

        with open("temp_2.pdf", "wb") as temp_pdf:
            temp_pdf.write(decoded_pdf2)

        # Si no ocurre ningún error, eliminamos los archivos temporales
        os.remove("temp_1.pdf")
        os.remove("temp_2.pdf")
    except Exception as e:
        return jsonify({'error': f"Error al decodificar el archivo: {str(e)}"}), 500

    try:
        # Conversión de los documentos a imágenes
        logging.debug("Iniciando conversión a imágenes.")
        image_1 = converter.convert_to_images(decoded_pdf1)
        image_2 = converter.convert_to_images(decoded_pdf2)
        logging.debug(f"Resultado de la conversión a imágenes: {image_1} - {image_2}")

        # Procesamiento de las imágenes a cadenas
        text_1 = producer.process_images(image_1)
        text_2 = producer.process_images(image_2)
        logging.debug(f"Resultado del procesamiento de imágenes a textos. {text_1} - {text_2}")

        # Tokenización y Lematización
        tokens_1 = tokenizer.tokenize_texts(text_1)
        tokens_2 = tokenizer.tokenize_texts(text_2)
        lems_1 = lemmatizer.lemmatizing_words(tokens_1)
        lems_2 = lemmatizer.lemmatizing_words(tokens_2)
        lems = [lems_1, lems_2]  # Pasamos a una lista todos los lemas para la vectorización

        # Vectorización y cálculos de similitud
        vectors = vectorizer.vectorize_doc(lems)
        similarity = vectorizer.similarity_docs(vectors)
        similarity_array = np.array(similarity)
        porcentaje_matrix = similarity_array[0, 1]
        porcentaje = porcentaje_matrix * 100

        # Si todo va bien, se procede a dar el código de estado 200 y un mensaje
        return handle_response(200, pdf1_filename, pdf2_filename, f"Subida y procesamiento de archivos completada.", f"{porcentaje:.2f}")

    except Exception as e:
        return jsonify({'error': f"Error en la conversión del archivo: {str(e)}"}), 500
    
@analize.route(service_get_historico, methods=['GET'])
def show_historico():
    if request.method != 'GET':
        return 405, "Error: método no soportado"
    
    data = DbManager.get_historico()
    return jsonify({'Peticiones': data})
    


#Función que maneja muestra los códigos de estado y su mensaje correspondiente
#Guardamos esto en la base de dato como histórico
def handle_response(estado, doc1, doc2, mensaje, similitud):
    DbManager.save_request(f"Solicitud exitosa. Código: {estado}", doc1, doc2, mensaje, similitud)
    return jsonify({"mensaje": mensaje, "Código": estado, "Similitud": f"{similitud}"}), estado

    


    


