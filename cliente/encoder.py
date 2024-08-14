
import base64
from io import BytesIO
import io
import logging
import PyPDF2


class Encoder:
    def __init__(self, file_path) -> None:
        self.file_path = file_path

    @staticmethod
    def encode_file_b64(file):
        file_content = file.read()  # Leemos el contenido del archivo
        encoded_file = base64.b64encode(file_content).decode("utf-8")
        return encoded_file

    @staticmethod
    def decode_file(encoded_file):
        if isinstance(encoded_file, str):
            encoded_file = encoded_file.encode('utf-8')
        missing_padding = len(encoded_file) % 4
        if missing_padding:
            encoded_file += b'=' * (4 - missing_padding)
        return base64.b64decode(encoded_file)
    
    @staticmethod
    def decode_file_b64(encoded_file):
        try:
            if isinstance(encoded_file, str):
                encoded_file = encoded_file.encode('utf-8')
            #Validamos el padding
            missing_padding = len(encoded_file) % 4
            if missing_padding:
                encoded_file += b'=' * (4 - missing_padding)

            #Decodificamos la cadena base64
            decoded_file = base64.b64decode(encoded_file, validate=True)
            return decoded_file
            
        except Exception as e:
            raise ValueError(f"Error al decodificar el archivo: {str(e)}")

    
    def validate_pdf(pdf_bytes):
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            num_pages = len(pdf_reader.pages)
            logging.debug(f"The PDF is valid with {num_pages} pages.")
            return True
        except PyPDF2.errors.PdfReadError as e:
            logging.error(f"PDF Read Error: {e}")
            return False

        
            
    
