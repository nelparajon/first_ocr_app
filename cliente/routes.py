import logging
import os
from flask import Flask, jsonify, request, Blueprint
from encoder import Encoder
from pdf_converter import PDFConverter
from ocr_producer import OCRProducer
from tokenizer import Tokenizer
from lemmatizer import Lemmatizer
from vectorizer import Vectorizer

# Configurar que el logging se guarde en un archivo
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

@analize.route('/')
def view():
    return '<h1> ANALIZADOR DE TEXTOS<h1>'

@analize.route('/analizar_documentos', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        
        data = request.get_json()

        logging.debug(f"Datos recibidos en la solicitud post: {data}")
        if 'file_1' not in data or 'file_2' not in data:
            return jsonify("Error con el archivo. No se han encontrado los dos documentos para comparar")
        
        #Extraemos cada documento en Base64
        encoded_file_1 = data['file_1']
        encoded_file_2 = data['file_2']
        
        try:
            #Decodificación
            decoded_file_1= Encoder.decode_file(encoded_file_1)
            decoded_file_2 = Encoder.decode_file(encoded_file_2)
            logging.debug("Archivo decodificado correctamente.")

            #Guardamos los archivos decodifidaos de manera temporal para el analisis de errores
            with open("temp_1.pdf", "wb") as temp_pdf:
                temp_pdf.write(decoded_file_1)
            
            with open("temp_2.pdf", "wb") as temp_pdf:
                temp_pdf.write(decoded_file_2)

            #Validación de que los documentos están en formato PDF
            if not Encoder.validate_pdf(decoded_file_1) or not Encoder.validate_pdf(decoded_file_2):
                return jsonify({"error": "Los archivos decodificados no parecen ser archivos PDF válidos."}), 400

            #Si no ocurre ningún error, eliminamos los archivos temporales
            os.remove("temp_1.pdf")
            os.remove("temp_2.pdf")
        except Exception as e:
            return jsonify({"error": f"Error al decodificar el archivo: {str(e)}"})

        try:
            #Conversión de los documentos a imágenes
            logging.debug("Iniciando conversión a imágenes.")
            image_1 = converter.convert_to_images(decoded_file_1)
            image_2 = converter.convert_to_images(decoded_file_2)
            logging.debug(f"Resultado de la conversión a imágenes: {image_1} - {image_2}")

            #Procesamiento de las imágenes a cadenas
            text_1 = producer.process_images(image_1)
            text_2 = producer.process_images(image_2)
            logging.debug(f"Resultado del procesamiento de imágenes a textos. {text_1} - {text_2}")

            #Tokenización y Lematización
            tokens_1 = tokenizer.tokenize_texts(text_1)
            tokens_2 = tokenizer.tokenize_texts(text_2)
            lems_1 = lemmatizer.lemmatizing_words(tokens_1)
            lems_2 = lemmatizer.lemmatizing_words(tokens_2)
            lems = [lems_1, lems_2] #Pasamos a una lista todos los lemas para la vectorización

            #Vectorización y calculos de similitud
            vectors = vectorizer.vectorize_doc(lems)
            similarity = vectorizer.similarity_docs(vectors)


        except Exception as e:
            return jsonify({"error": f"Error en la conversión del archivo: {str(e)}"})

        return jsonify(f"Subida y procesamiento de archivos completada."
                       f"Procesamiento de imágenes completado. TOKENS ARCHIVO 1: {tokens_1}\n - TOKENS ARCHIVO 2: {tokens_2} \n Similitud {similarity}")


    return jsonify("Método no soportado")




