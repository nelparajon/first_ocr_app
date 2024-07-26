
import base64
from io import BytesIO
import io
import logging
import PyPDF2


class Encoder:
    def __init__(self, file_path) -> None:
        self.file_path = file_path

    @staticmethod
    def encode_file_b64(file_path):
        with open(file_path, "rb") as file:
            encoded_file = base64.b64encode(file.read()).decode("utf-8")
            return encoded_file

    @staticmethod
    def decode_file(encoded_file):
        missing_padding = len(encoded_file) % 4
        if missing_padding:
            encoded_file += '=' * (4 - missing_padding)
        return base64.b64decode(encoded_file)

    def validate_pdf(pdf_bytes):
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            num_pages = len(pdf_reader.pages)
            logging.debug(f"The PDF is valid with {num_pages} pages.")
            return True
        except PyPDF2.errors.PdfReadError as e:
            logging.error(f"PDF Read Error: {e}")
            return False

        
            
    
