import base64
import logging
import os
from flask import Flask, jsonify, request, Blueprint
import numpy as np
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
upload_files_b64 = os.getenv('UPLOAD_FILES_B64')


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

        #Imprimir para depuración
        print("Base64 de PDF1:", pdf1_base64[:-100])
        print("Base64 de PDF2:", pdf2_base64[:-100])

        # Decodificar los archivos base64
        decoded_pdf1 = Encoder.decode_file(pdf1_base64)
        decoded_pdf2 = Encoder.decode_file(pdf2_base64)

        # Validar que los archivos sean PDFs
        if not Encoder.validate_pdf(decoded_pdf1) or not Encoder.validate_pdf(decoded_pdf2):
            return handle_response(422, pdf1_filename, pdf2_filename, f"Error al validar el formato de los archivos", 0.0)

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
    
@analize.route('/historico', methods=['GET'])
def show_historico():
    if request.method != 'GET':
        return 405, "Error: método no soportado"

    try:
        data = DbManager.get_historico()
        return jsonify({'Peticiones': data})
    except Exception as ex:
        raise ex

@analize.route("/analizar_documentos_b64", methods=['POST'])
def analizar_documentos_base64():
    data = request.get_json()

    if 'file1' not in data or 'file2' not in data:
        return handle_response(400, None, None, f"Se necesitan dos archivos para comparar", 0.0)
    
    file1_base64 = data['file1'].get('content')
    file2_base64 = data['file2'].get('content')
    file1_name = data['file1'].get('name')
    file2_name= data['file2'].get('name')

    if not file1_base64 or not file2_base64:
         return handle_response(400, file1_name, file2_name, f"El contenido de los archivos no puede estar vacío", 0.0)

    try:
        print("Se procede a decodificar los archivos...")
        #decodificar los archivos
        decoded_file1 = base64.b64decode(file1_base64, validate=True)
        decoded_file2 = base64.b64decode(file2_base64, validate=True)
        

        if not Encoder.validate_pdf(decoded_file1) or not Encoder.validate_pdf(decoded_file2):
            return handle_response(405, file1_name, file2_name, f"Error al validar el formato de los archivos", 0.0)
        
    except ValueError as e:
        return jsonify({'error': f"Error al decodificar el archivo: {str(e)}"}), 400    
    except Exception as e:
        return jsonify({'error': f"Error al decodificddsar el archivo: {str(e)}"}), 500
    
    
    try:
        # Conversión de los documentos a imágenes
        logging.debug("Iniciando conversión a imágenes.")
        image_1 = converter.convert_to_images(decoded_file1)
        image_2 = converter.convert_to_images(decoded_file2)
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
        return handle_response(200, file1_name, file2_name, f"Subida y procesamiento de archivos completada.", f"{porcentaje:.2f}")
    except Exception as e:
        return jsonify({'error': f"Error en la conversión del archivo: {str(e)}"}), 500

#Función que maneja y muestra los códigos de estado y su mensaje correspondiente
#Guardamos esto en la base de datos en la tabla histórico
def handle_response(estado, doc1, doc2, mensaje, similitud):
    #Verifica que los pdf no son none
    if doc1 is None or doc2 is None:
        raise ValueError("doc1 y doc2 no pueden ser None.")
    
    DbManager.save_request(f"Solicitud exitosa. Código: {estado}", doc1, doc2, mensaje, similitud)
    return jsonify({"mensaje": mensaje, "Código": estado, "Similitud": f"{similitud}"}), estado



    


