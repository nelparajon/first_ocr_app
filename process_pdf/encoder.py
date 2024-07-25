import base64
from io import BytesIO


class Encoder:
    def __init__(self, file_path) -> None:
        self.file_path = file_path

    def encode_file_b64(self):
        with open(self.file_path, "rb") as file:
            encode_file = base64.b64encode(file.read()).decode("utf-8")  # utf-8 para que sea legible en json
            print(encode_file)
        return encode_file

    
    def decode_file(encoded_file):
        
        missing_padding = len(encoded_file) % 4
        if missing_padding:
            encoded_file += '=' * (4 - missing_padding)
        return  base64.b64decode(encoded_file)
    
    

        
            
    
